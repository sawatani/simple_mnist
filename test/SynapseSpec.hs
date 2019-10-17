module SynapseSpec (spec) where

import qualified Numeric.LinearAlgebra as A
import           Synapse
import           Test.Hspec
import           Test.Hspec.QuickCheck (prop)
import           Test.QuickCheck
import           Text.Printf

runTest :: IO ()
runTest = hspec spec

spec :: Spec
spec = do
  describe "Double" $ do
    prop "multiple" $
      let small a = a /= 0 && -1 < a && a < 1
       in forAll (vectorOf 3 (genSignal `suchThat` small)) $ \[a, b, c] ->
            a * b * c `shouldNotBe` 0
  describe "Sigmoid" propsSigmoid
  describe "ReLU" propsReLU
  describe "Softmax" propsSoftmax
  describe "CrossEntropy" propsCrossEntropy

propsSigmoid = do
  prop "make with exp" $
    forAll genSignal $ \n ->
      let r = sigmoid n
          x = 1.0 / (1.0 + exp (-n))
       in r `shouldBe` x
  prop "apply all elements" $
    forAll genMN $ \(m, n) ->
      forAll (genSignals 1 (m * n)) $ \[vs] ->
        let r = sigmoidm $ (m A.>< n) vs
            rs = A.toList `concatMap` A.toRows r
            xs = sigmoid `map` vs
         in rs `shouldBe` xs
  prop "calc diff" $
    forAll genMN $ \(m, n) ->
      forAll (genSignals 2 (m * n)) $ \[ys, ds] ->
        let r = sigmoidBackward ((m A.>< n) ys) ((m A.>< n) ds)
            rs = A.toList `concatMap` A.toRows r
            xs = zipWith (\y d -> d * (1.0 - y) * y) ys ds
         in rs `shouldBe` xs

propsReLU = do
  prop "take max" $
    forAll genSignal $ \v ->
      let r = relu v
          x = 0 `max` v
       in r `shouldBe` x
  prop "apply all elements" $
    forAll genMN $ \(m, n) ->
      forAll (genSignals 1 (m * n)) $ \[vs] ->
        let r = relum $ (m A.>< n) vs
            rs = A.toList `concatMap` A.toRows r
            xs = relu `map` vs
         in rs `shouldBe` xs
  prop "calc diff" $
    forAll genMN $ \(m, n) ->
      forAll (genSignals 2 (m * n)) $ \[ys, ds] ->
        let r = relumBackward ((m A.>< n) ys) ((m A.>< n) ds)
            rs = A.toList `concatMap` A.toRows r
            reverse x =
              if 0 < x
                then 1
                else 0
            xs = zipWith (\x d -> d * fromIntegral (reverse x)) ys ds
         in rs `shouldBe` xs

propsSoftmax = do
  prop "take max" $
    forAll (listOf1 genSignal) $ \vs ->
      let r = softmax $ A.vector vs
          c = maximum vs
          ds = map (\v -> exp $ v - c) vs
          s = sum ds
          xs = map (/ s) ds
       in (A.toList r) `shouldBe` xs
  prop "apply all rows" $
    forAll genMN $ \(m, n) ->
      forAll (genSignals m n) $ \vss ->
        let r = softmaxm $ (m A.>< n) $ concat vss
            xs = map (softmax . A.vector) vss
         in (A.toRows r) `shouldBe` xs

propsCrossEntropy = do
  prop "take sum" $
    forAll (listOf1 genSignal) $ \vs ->
      forAll (genTeacher $ length vs) $ \ts ->
        let ys = softmax $ A.vector vs
            d = 1e-10
            r = crossEntropy (A.vector ts) ys
            cs = zipWith (\t y -> t * log (y + d)) ts $ A.toList ys
            x = -sum cs
         in r `shouldBe` x
  prop "apply all rows" $
    forAll genMN $ \(m, n) ->
      forAll (genSignals m n) $ \vss ->
        forAll (vectorOf m $ genTeacher n) $ \tss ->
          let ys = map (softmax . A.vector) vss
              ts = A.vector `map` tss
              r = crossEntropym (A.fromRows ts) (A.fromRows ys)
              xs = zipWith crossEntropy ts ys
              x = sum xs / fromIntegral m
           in r `shouldBe` x

genSignal :: Gen Double
genSignal = (arbitrary :: Gen Double) `suchThat` (\a -> -10 <= a && a <= 10)

genSignals :: Int -> Int -> Gen [[Double]]
genSignals m n = vectorOf m (vectorOf n genSignal)

genMN :: Gen (Int, Int)
genMN = do
  m <- choose (1, 10)
  n <- choose (1, 10)
  return (m, n)

genTeacher :: Int -> Gen [Double]
genTeacher n = do
  i <- choose (0, n - 1)
  return $ replicate i 0.0 ++ 1 : replicate (n - i -1) 0.0
