{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds   #-}

module Synapse
  ( sigmoid
  , sigmoidm
  , sigmoidBackward
  , relu
  , relum
  , relumBackward
  , softmax
  , softmaxm
  , softmaxWithCross
  , softmaxWithCrossBackward
  , crossEntropy
  , crossEntropym
  , affinem
  , affinemBackward
  ) where

import           Debug.Trace
import           Numeric.LinearAlgebra
import           Prelude               hiding ((<>))

sigmoid :: (Floating a) => a -> a
sigmoid a = 1 / (1 + exp (-a))

sigmoidm :: (Floating a, Container Matrix a) => Matrix a -> Matrix a
sigmoidm = cmap sigmoid

sigmoidBackward ::
     (Floating a, Num (Vector a), Container Matrix a)
  => Matrix a
  -> Matrix a
  -> Matrix a
sigmoidBackward y d = d * (1 - y) * y

relu :: (Ord a, Num a) => a -> a
relu = max 0

relum :: (Ord a, Num a, Container Matrix a) => Matrix a -> Matrix a
relum = cmap relu

relumBackward ::
     (Ord a, Num a, Num (Matrix a), Container Vector a, Container Matrix a)
  => Matrix a
  -> Matrix a
  -> Matrix a
relumBackward x d = d * mask
  where
    mask = cmap (fromIntegral . onoff) x
    onoff a
      | 0 < a = 1
      | otherwise = 0

softmax :: (Floating a, Container Vector a) => Vector a -> Vector a
softmax v = cmap (/ s) v'
  where
    m = maxElement v
    v' = cmap (\a -> exp (a - m)) v
    s = sumElements v'

softmaxm :: (Floating a, Container Vector a) => Matrix a -> Matrix a
softmaxm m = fromRows $ map softmax $ toRows m

crossEntropy ::
     (Floating a, Num (Vector a), Container Vector a)
  => Vector a
  -> Vector a
  -> a
crossEntropy t y = -(sumElements $ cmap log (y + d) * t)
  where
    d = 1e-10

crossEntropym ::
     (Floating a, Num (Vector a), Container Vector a)
  => Matrix a
  -> Matrix a
  -> a
crossEntropym t m = sum vs / batchSize
  where
    vs = zipWith crossEntropy (toRows t) (toRows m)
    batchSize = fromIntegral $ rows t

softmaxWithCross ::
     (Floating a, Num (Vector a), Container Vector a)
  => Matrix a
  -> Matrix a
  -> (Matrix a, a)
softmaxWithCross t x = (y, crossEntropym t y)
  where
    y = softmaxm x

softmaxWithCrossBackward ::
     (Floating a, Num (Vector a), Container Vector a)
  => Matrix a
  -> Matrix a
  -> Matrix a
softmaxWithCrossBackward t y = cmap (/ batchSize) $ y - t
  where
    batchSize = fromIntegral $ rows t

affinem ::
     (Floating a, Numeric a, Num (Vector a))
  => Matrix a
  -> Vector a
  -> Matrix a
  -> Matrix a
affinem w b x = (x <> w) + b'
  where
    b' = fromRows $ replicate (rows x) b

affinemBackward ::
     (Floating a, Numeric a)
  => Matrix a
  -> Matrix a
  -> Matrix a
  -> (Matrix a, Matrix a, Vector a)
affinemBackward w x d = (dx, dw, db)
  where
    dx = d <> tr w
    dw = tr x <> d
    db = fromList $ sumElements `map` toColumns d

batchNorm :: (Floating a, Numeric a, Num (Vector a)) => Vector a -> Vector a
batchNorm v = cmap f v
  where
    f x = (x - average) / b
    b = sqrt vari + small
    vari = sumElements ((v - scalar average) ^ 2) / m
    average = sumElements v / m
    m = fromIntegral $ size v
    small = 1e-10

batchNormm :: (Floating a, Numeric a, Num (Vector a)) => Matrix a -> Matrix a
batchNormm = fromRows . map batchNorm . toRows

batchNormmBackward ::
     ( Floating a
     , Numeric a
     , Num (Vector a)
     , Floating (Vector a)
     , Container Vector a
     )
  => Matrix a
  -> Matrix a
  -> Matrix a
  -> Matrix a
  -> Matrix a
  -> Vector a
  -> Vector a
  -> Matrix a
batchNormmBackward dout xhat gamma ivar xmu sqrtvar var = dx
  where
    (nRows, nCols) = size dout
    small = 1e-10
    sum0 m = fromList $ sumElements `map` toColumns m
    spawnRows r = fromRows $ replicate nRows (r / fromIntegral nRows)
    -- step 9
    beta = sum0 dout
    -- step 8
    dgamma = sum0 $ dout * xhat
    dxhat = dout * gamma
    -- step 7
    dxmu1 = dxhat * ivar
    divar = sum0 $ dxhat * xmu
    -- step 6
    dsqrtvar = (-1 / (sqrtvar ^ 2)) * divar
    -- step 5
    dvar = (1 / sqrt (var + small)) * dsqrtvar * 0.5
    -- step 4
    dsq = spawnRows dvar
    -- step 3
    dxmu2 = xmu * dsq * 2
    -- step 2
    dx1 = dxmu1 + dxmu2
    dmu = negate `cmap` sum0 dx1
    -- step 1
    dx2 = spawnRows dmu
    -- step 0
    dx = dx1 + dx2
