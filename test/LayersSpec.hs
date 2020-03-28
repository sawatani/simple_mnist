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
import           Numeric.LinearAlgebra   as A
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
  describe "Affine" propsAffine
  describe "Sigmoid" propsSigmoid
  describe "ReLU" propsReLU
  describe "Joined" propsJoined
  describe "Matrix" propsMatrix

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

propsJoined = do
  prop "forward" $
    forAll
      (do a <- choose (10, 100)
          b <- choose (10, 100)
          c <- choose (10, 100)
          f <- genAffine a b
          x <- genMatrix c a
          return (f, x)) $ \(fL1, x) ->
      let fL2 = SigmoidForward
          fJ = fL1 ~> fL2
          (JoinedBackwardLayer bL1 bL2, y) = forward fJ x
          (bL1', yL1) = forward fL1 x
          (bL2', yL2) = forward fL2 yL1
       in (bL1, bL2, y) `shouldBe` (bL1', bL2', yL2)
  prop "backward" $
    forAll
      (do a <- choose (10, 100)
          b <- choose (10, 100)
          c <- choose (10, 100)
          d <- genMatrix c b
          y <- genMatrix c b
          x <- genMatrix c a
          fL1 <- genAffine a b
          let AffineForward weights bias = fL1
          let bL1 = AffineBackward weights bias x
          return (bL1, SigmoidBackward y, d)) $ \(bL1, bL2, d) ->
      let bJ = bL1 <~ bL2
          (JoinedForwardLayer fL1 SigmoidForward, bD) = backward 1.0 bJ d
          (SigmoidForward, dL2) = backward 1.0 bL2 d
          (fL1', dL1) = backward 1.0 bL1 dL2
       in (fL1, bD) `shouldBe` (fL1', dL1)

propsMatrix = do
  prop "scale" $
    forAll genElements $ \(a, b, x, y) ->
      let m = b `A.scale` toMatrix x y a
          m' = toMatrix x y (a * b)
       in m `shouldBe` m'
  prop "multiply" $
    forAll genElements $ \(a, b, x, y) ->
      let m = toMatrix x y b * toMatrix x y a
          m' = toMatrix x y (a * b)
       in m `shouldBe` m'
  prop "subtract" $
    forAll genElements $ \(a, b, x, y) ->
      let m = toMatrix x y a - toMatrix x y b
          m' = toMatrix x y (a - b)
       in m `shouldBe` m'
  prop "add" $
    forAll genElements $ \(a, b, x, y) ->
      let m = toMatrix x y a + toMatrix x y b
          m' = toMatrix x y (a + b)
       in m `shouldBe` m'
  where
    toMatrix x y a = fromLists $ replicate x $ replicate y a
    genElements = do
      a <- arbitrary
      b <- arbitrary
      x <- choose (10, 100)
      y <- choose (10, 100)
      return (a :: R, b, x, y)

genVectorN :: Int -> Gen (Vector R)
genVectorN n = fmap fromList $ vectorOf n $ choose (1, 100)

genAffine :: Int -> Int -> Gen (ForwardLayer R)
genAffine nIn nOut = do
  weights <- fmap fromLists $ vectorOf nIn $ vectorOf nOut arbitrary
  bias <- fromList <$> vectorOf nOut arbitrary
  return $ AffineForward weights bias

genMatrix :: Int -> Int -> Gen (Matrix R)
genMatrix nRows nCols =
  fmap fromLists $ vectorOf nRows $ vectorOf nCols arbitrary
