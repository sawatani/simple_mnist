{-# LANGUAGE FlexibleContexts #-}

module LearnSpec
  ( spec
  ) where

import           Control.Monad
import qualified Data.Set                as Set
import           Data.Tuple.Extra
import           Debug.Trace
import           Foundation.Monad
import           Layers
import           Learn
import           Mnist
import           Numeric.LinearAlgebra
import           Test.Hspec
import           Test.Hspec.QuickCheck   (prop)
import           Test.QuickCheck
import           Test.QuickCheck.Gen
import           Test.QuickCheck.Monadic

runTest :: IO ()
runTest = hspec spec

spec :: Spec
spec = do
  describe "hotone" propsHotone
  describe "convertTests" propsConvertTests
  describe "convertTrains" propsConvertTrains
  describe "initNN" propsInitNN

propsHotone =
  prop "length" $
  forAll (genNM 100) $ \(n, m) ->
    let v = toList $ hotone n m :: [R]
        (a, k:c) = splitAt m v
        x = sum $ a ++ c
     in (length v, x, k) `shouldBe` (n, 0, 1)

propsConvertTests =
  prop "div 255" $
  forAll genMnistData $ \d ->
    let r = convertTests d
        MnistData ns = d
        (ts, is) = second concat $ unzip $ map (second $ concat . toLists) ns
        (ps, vs) = second concat $ unzip $ map (second toList) r
        x1 = map fromIntegral ts
        x2 = map (round . (* 255)) vs
     in (x1, is) `shouldBe` (ps, x2)

propsConvertTrains = do
  prop "length" $
    forAll genMnistData $ \d ->
      let r = convertTrains (length ns `div` 10) d
          MnistData ns = d
          (xs, ys) = unzip $ map countRows r
          countRows (TrainBatch a) = both rows a
          a = sum xs
          b = length ns
       in (xs, a) `shouldBe` (ys, b)
  prop "size" $
    forAll
      (do d <- genMnistData
          s <- choose (2, 10)
          return (s, d)) $ \(s, d) ->
      let r = convertTrains s d
          MnistData ns = d
          (w, c) = size $ snd $ head ns
          xs = unzip $ map getSize r
          getSize (TrainBatch a) = both size a
          (a, b) = both (maximum . Set.toList . Set.fromList) xs
       in (a, b) `shouldBe` ((s, 10), (s, w * c))
  prop "div 255" $
    forAll genMnistData $ \d ->
      let r = convertTrains (length ns `div` 10) d
          MnistData ns = d
          xs = map fromIntegral $ concatMap (concat . toLists . snd) ns
          ys = map (round . (* 255)) $ concatMap flatMs r
          flatMs (TrainBatch (_, ms)) = concat $ toLists ms
       in xs `shouldBe` ys

propsInitNN =
  it "size of layers" $ do
    r <- initNN ReLUForward [16, 50, 8, 10]
    let ForwardNN lA SoftmaxWithCrossForward = r
    let JoinedForwardLayer lB (AffineForward m3 b3) = lA
    let JoinedForwardLayer lC ReLUForward = lB
    let JoinedForwardLayer lD (AffineForward m2 b2) = lC
    let JoinedForwardLayer (AffineForward m1 b1) ReLUForward = lD
    let ms = map size [m1, m2, m3]
    let bs = map size [b1, b2, b3]
    (ms, bs) `shouldBe` ([(16, 50), (50, 8), (8, 10)], [50, 8, 10])

genNM :: Int -> Gen (Int, Int)
genNM x = do
  n <- choose (1, x)
  m <- choose (0, n - 1)
  return (n, m)

genMnistData :: Gen MnistData
genMnistData = do
  len <- choose (10, 100)
  nRows <- choose (10, 20)
  nCols <- choose (10, 20)
  ms <- vectorOf len $ genMatrixZ 255 nRows nCols
  vs <- vectorOf len $ choose (0, 9)
  return $ MnistData $ zip vs ms

genMatrixZ :: Int -> Int -> Int -> Gen (Matrix Z)
genMatrixZ value nRows nCols = do
  zss <- vectorOf nRows $ vectorOf nCols $ fromIntegral <$> choose (0, value)
  return $ fromLists zss
