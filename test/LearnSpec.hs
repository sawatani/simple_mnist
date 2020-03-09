module LearnSpec (spec) where

import           Learn
import           Mnist
import Foundation.Monad
import Control.Monad
import Debug.Trace
import           Data.Bifunctor
import           Numeric.LinearAlgebra
import           Test.Hspec
import           Test.Hspec.QuickCheck (prop)
import           Test.QuickCheck
import           Test.QuickCheck.Gen
import           Test.QuickCheck.Monadic

runTest :: IO ()
runTest = hspec spec

spec :: Spec
spec = do
  describe "hotone" propsHotone
  describe "convertTests" propsConvertTests

propsHotone =
  prop "length" $ forAll (genNM 100) $ \(n, m) ->
  let v = toList $ hotone n m :: [R]
      (a, k:c) = splitAt m v
      x = sum $ a ++ c
   in (length v, x, k) `shouldBe` (n, 0, 1)

propsConvertTests =
  prop "div 255" $ forAll genMnistData $ \d ->
  let r = convertTests d
      MnistData ns = d
      (ts, is) = second concat $ unzip $ map (second $ concat . toLists) ns
      (ps, vs) = second concat $ unzip $ map (second toList) r
      x1 = map fromIntegral ts
      x2 = map (round . (*255)) vs
  in (x1, is) `shouldBe` (ps, x2)

genNM :: Int -> Gen (Int, Int)
genNM x = do
  n <- choose (1, x)
  m <- choose (0, n - 1)
  return (n, m)

genMnistData :: Gen MnistData
genMnistData = do
  len <- choose (1, 10)
  nRows <- choose (1, 10)
  nCols <- choose (1, 10)
  ms <- vectorOf len $ genMatrixZ 255 nRows nCols
  vs <- vectorOf len $ choose (0, 10)
  return $ MnistData $ zip vs ms

genMatrixZ :: Int -> Int -> Int -> Gen (Matrix Z)
genMatrixZ value nRows nCols = do
  zss <- vectorOf nRows $ vectorOf nCols $ fromIntegral <$> choose (0, value)
  return $ fromLists zss
