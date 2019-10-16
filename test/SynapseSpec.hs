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
      forAll (vectorOf 3 (genSignal `suchThat` (/= 0))) $ \[a, b, c] ->
        (a * b * c /= 0) `shouldBe` True
  describe "Sigmoid" $ do
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
  describe "ReLU" $ do
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

genSignal :: Gen Double
genSignal = (arbitrary :: Gen Double) `suchThat` (\a -> -10 <= a && a <= 10)

genSignals :: Int -> Int -> Gen [[Double]]
genSignals m n = vectorOf m (vectorOf n genSignal)

genMN :: Gen (Int, Int)
genMN = do
  m <- choose (1, 10)
  n <- choose (1, 10)
  return (m, n)
