module SynapseSpec
  ( spec
  ) where

import qualified Numeric.LinearAlgebra as A
import           Synapse
import           Test.Hspec
import           Test.Hspec.QuickCheck (prop)
import           Test.QuickCheck

runTest :: IO ()
runTest = hspec spec

spec :: Spec
spec = do
  describe "Double" propsDouble
  describe "Sigmoid" propsSigmoid
  describe "ReLU" propsReLU
  describe "Softmax" propsSoftmax
  describe "CrossEntropy" propsCrossEntropy
  describe "SoftmaxWithCross" propsSoftmaxWithCross
  describe "Affine" propsAffine

propsDouble = do
  prop "multiple" $
    forAll (vectorOf 3 (genSignal `suchThat` small)) $ \[a, b, c] ->
      let x3 a = a ^ 3
       in x3 a * x3 b * x3 c `shouldNotBe` 0
  prop "roundy" $
    forAll (genSignal `suchThat` small) $ \a ->
      let b = abs a
          x = foldr (\i j -> b ^ i + j) 0 [0 .. 6]
          y = roundy 4 x
          e = 0.0001
       in (y < x && (x - e) < y) `shouldBe` True
  where
    small a = a /= 0 && -1 < a && a < 1

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

propsSoftmaxWithCross = do
  prop "composite" $
    forAll genMN $ \(m, n) ->
      forAll (genSignals m n) $ \vss ->
        forAll (vectorOf m $ genTeacher n) $ \tss ->
          let tx = A.fromRows $ map A.vector tss
              vx = A.fromRows $ map A.vector vss
              r = softmaxWithCross tx vx
              y = softmaxm vx
              c = crossEntropym tx y
           in r `shouldBe` (y, c)
  prop "calc diff" $
    forAll genMN $ \(m, n) ->
      forAll (genSignals m n) $ \vss ->
        forAll (vectorOf m $ genTeacher n) $ \tss ->
          let ts = concat tss
              tx = (m A.>< n) ts
              y = softmaxm $ A.fromLists vss
              ys = concat $ A.toLists y
              r = softmaxWithCrossBackward tx y
              d = (m A.>< n) $ zipWith (\y t -> (y - t) / fromIntegral m) ys ts
           in reduce r `shouldBe` reduce d

propsAffine = do
  prop "forward" $
    forAll (choose (1, 10)) $ \l ->
      forAll genMN $ \(m, n) ->
        forAll (genSignals m n) $ \wss ->
          forAll (vectorOf n genSignal) $ \bs ->
            forAll (genSignals l m) $ \xss ->
              let x = A.fromLists xss
                  w = A.fromLists wss
                  b = A.vector bs
                  y = affinem w b x
                  rows = zipWith row (A.toRows x) (replicate l $ A.toColumns w)
                  r = A.fromLists $ map (zipWith (+) bs) rows
               in reduce y `shouldBe` reduce r
  prop "backward" $
    forAll (choose (1, 10)) $ \l ->
      forAll genMN $ \(m, n) ->
        forAll (genSignals m n) $ \wss ->
          forAll (genSignals l m) $ \xss ->
            forAll (genSignals l n) $ \dss ->
              let x = A.fromLists xss
                  w = A.fromLists wss
                  d = A.fromLists dss
                  (nx, nw, nb) = affinemBackward w x d
                  dx =
                    A.fromLists $
                    zipWith row (A.toRows d) (replicate l $ A.toRows w)
                  dw =
                    A.fromLists $
                    zipWith row (A.toColumns x) (replicate m $ A.toColumns d)
                  db = A.vector $ foldr (zipWith (+)) (replicate n 0.0) dss
               in (reduce nx, reduce nw, reduce $ A.asRow nb) `shouldBe`
                  (reduce dx, reduce dw, reduce $ A.asRow db)

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
  return $ replicate i 0.0 ++ 1 : replicate (n - i - 1) 0.0

roundy n v = fromIntegral (floor $ v * 10 ^ n) / 10.0 ^^ n

reduce a = map (roundy 8) $ concat $ A.toLists a

cell arow bcol = sum $ zipWith (*) (A.toList arow) (A.toList bcol)

row arow = map (cell arow)
