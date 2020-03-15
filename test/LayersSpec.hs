module LayersSpec
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
import           Numeric.LinearAlgebra as A
import           Synapse
import           Test.Hspec
import           Test.Hspec.QuickCheck   (prop)
import           Test.QuickCheck
import           Test.QuickCheck.Gen
import           Test.QuickCheck.Monadic

runTest :: IO ()
runTest = hspec spec

spec :: Spec
spec = do
  describe "predict" propsPredict
  describe "evaluate" propsEvaluate
  describe "Affine" propsAffine
  describe "Sigmoid" propsSigmoid
  describe "ReLU" propsReLU

propsPredict =
  prop "index" $
  forAll (genVectorN 100) $ \v ->
    let r = predict ReLUForward v
        a = maxIndex v
     in r `shouldBe` a

propsEvaluate = do
  prop "100%" $
    forAll (vectorOf 100 $ genVectorN 100) $ \vs ->
      let ns = map maxIndex vs
          r = evaluate ReLUForward $ zip ns vs
       in r `shouldBe` 1
  prop "0%" $
    forAll (vectorOf 100 $ genVectorN 100) $ \vs ->
      let ns = map ((+1) . maxIndex) vs
          r = evaluate ReLUForward $ zip ns vs
       in r `shouldBe` 0
  prop "50%" $
    forAll (vectorOf 100 $ genVectorN 100) $ \vs ->
      let (vA, vB) = splitAt 50 vs
          nA = map maxIndex vA
          nB = map ((+1) . maxIndex) vB
          ns = nA ++ nB
          r = evaluate ReLUForward $ zip ns vs
       in r `shouldBe` 0.5

propsAffine = do
  prop "forward" $
    forAll
      (do a <- choose (10, 100)
          b <- choose (10, 100)
          c <- choose (10, 100)
          f <- genAffine a b
          s <- genMatrix c a
          return (f, s)) $ \(af, x) ->
      let AffineForward weights bias = af
          (AffineBackward rW rB rX, r) = forward af x
          e = affinem weights bias x
       in (rW, rB, rX, r) `shouldBe` (weights, bias, x, e)
  prop "backward" $
    forAll
      (do a <- choose (10, 100)
          b <- choose (10, 100)
          c <- choose (10, 100)
          d <- genMatrix c b
          f <- genAffine a b
          s <- genMatrix c a
          return (f, s, d)) $ \(af, x, d) ->
      let AffineForward w b = af
          rate = 1.0
          (AffineForward w' b', d') = backward rate (AffineBackward w b x) d
          (dx, dw, db) = affinemBackward w x d
          rW = w - dw
          rB = b - db
       in (w', b', d') `shouldBe` (rW, rB, dx)
  prop "backward with rate" $
    forAll
      (do a <- choose (10, 100)
          b <- choose (10, 100)
          c <- choose (10, 100)
          r <- arbitrary `suchThat` (\a -> 0 < a && a < 1)
          d <- genMatrix c b
          f <- genAffine a b
          s <- genMatrix c a
          return (r, f, s, d)) $ \(rate, af, x, d) ->
      let AffineForward w b = af
          (AffineForward w' b', d') = backward rate (AffineBackward w b x) d
          (dx, dw, db) = affinemBackward w x d
          rW = w - A.scale rate dw
          rB = b - A.scale rate db
       in (w', b', d') `shouldBe` (rW, rB, dx)

propsSigmoid = do
  prop "forward" $
    forAll
      (do a <- choose (10, 100)
          b <- choose (10, 100)
          genMatrix a b) $ \x ->
      let (SigmoidBackward y, r) = forward SigmoidForward x
          e = sigmoidm x
       in (y, r) `shouldBe` (r, e)
  prop "backward" $
    forAll
      (do a <- choose (10, 100)
          b <- choose (10, 100)
          d <- genMatrix a b
          y <- genMatrix a b
          return (y, d)) $ \(y, d) ->
      let (SigmoidForward, d') = backward 0 (SigmoidBackward y) d
          e = sigmoidBackward y d
       in d' `shouldBe` e

propsReLU = do
  prop "forward" $
    forAll
      (do a <- choose (10, 100)
          b <- choose (10, 100)
          genMatrix a b) $ \x ->
      let (ReLUBackward y, r) = forward ReLUForward x
          e = relum x
       in (y, r) `shouldBe` (x, e)
  prop "backward" $
    forAll
      (do a <- choose (10, 100)
          b <- choose (10, 100)
          x <- genMatrix a b
          d <- genMatrix a b
          return (x, d)) $ \(x, d) ->
      let (ReLUForward, d') = backward 0 (ReLUBackward x) d
          e = relumBackward x d
       in d' `shouldBe` e

genVectorN :: Int -> Gen (Vector R)
genVectorN n = fmap fromList $ vectorOf n $ choose (1, 100)

genAffine :: Int -> Int -> Gen (ForwardLayer R)
genAffine nIn nOut = do
  weights <- fmap fromLists $ vectorOf nIn $ vectorOf nOut arbitrary
  bias <- fromList <$> vectorOf nOut arbitrary
  return $ AffineForward weights bias

genMatrix :: Int -> Int -> Gen (Matrix R)
genMatrix nRows nCols = fmap fromLists $ vectorOf nRows $ vectorOf nCols arbitrary
