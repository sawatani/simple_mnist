module Synapse
    (
      sigmoid
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

import           Prelude                     hiding ((<>))
import           Debug.Trace
import           Numeric.LinearAlgebra

sigmoid :: (Floating a) => a -> a
sigmoid a = fromIntegral 1 / (fromIntegral 1 + exp (-a))

sigmoidm :: (Floating a, Container Matrix a) => Matrix a -> Matrix a
sigmoidm = cmap sigmoid

sigmoidBackward :: (Floating a, Num (Vector a), Container Matrix a) => Matrix a -> Matrix a -> Matrix a
sigmoidBackward y d = d * (1 - y) * y

relu :: (Ord a, Num a) => a -> a
relu = max $ fromIntegral 0

relum :: (Ord a, Num a, Container Matrix a) => Matrix a -> Matrix a
relum = cmap relu

relumBackward :: (Ord a, Num a, Num (Matrix a), Container Vector a, Container Matrix a) => Matrix a -> Matrix a -> Matrix a
relumBackward x d = d * mask
    where mask = cmap (\a -> fromIntegral $ if (0 < a) then 1 else 0) x

softmax :: (Floating a, Container Vector a) => Vector a -> Vector a
softmax v = cmap (/s) v'
    where
        m = maxElement v
        v' = cmap (\a -> exp (a - m)) v
        s = sumElements v'

softmaxm :: (Floating a, Container Vector a) => Matrix a -> Matrix a
softmaxm m = fromRows $ map softmax $ toRows m

crossEntropy :: (Floating a, Num (Vector a), Container Vector a) => Vector a -> Vector a -> a
crossEntropy t y = -(sumElements $ cmap log (y + d) * t)
    where d = 1e-10

crossEntropym :: (Floating a, Num (Vector a), Container Vector a) => Matrix a -> Matrix a -> a
crossEntropym t m = sum vs / batchSize
    where
        vs = uncurry crossEntropy `map` (toRows t `zip` toRows m)
        batchSize = fromIntegral $ rows t

softmaxWithCross :: (Floating a, Num (Vector a), Container Vector a) => Matrix a -> Matrix a -> (Matrix a, a)
softmaxWithCross t x = (y, crossEntropym t y)
    where
        y = softmaxm x

softmaxWithCrossBackward :: (Floating a, Num (Vector a), Container Vector a, Show a) => Matrix a -> Matrix a -> Matrix a
softmaxWithCrossBackward t y = (y - t) / batchSize
    where
        batchSize = fromIntegral $ rows t

affinem :: (Floating a, Numeric a, Num (Vector a), Show a) => Matrix a -> Vector a -> Matrix a -> Matrix a
affinem w b x = trace (
        "-------------"
        ++ "\nAffine input: " ++ show (sumElements x)
        ++ "\nAffile weight: " ++ show (sumElements w)
        ++ "\nAffine bias: " ++ show (sumElements b')
        ++ "\nAffine result: " ++ show (sumElements r)
        ) r
    where
        r = (x <> w) + b'
        b' = fromRows $ replicate (rows x) b

affinemBackward :: (Floating a, Numeric a, Show a) => Matrix a -> Matrix a -> Matrix a -> (Matrix a, Matrix a, Vector a)
affinemBackward w x d = trace (
        "______________"
        ++ "\n AffineBackword weight: " ++ show (sumElements dw)
        ++ "\n AffineBackword bias: " ++ show (sumElements db)
        ++ "\n AffineBackword backword: " ++ show (sumElements dx)
        ) (dx, dw, db)
    where
        dx = d <> tr w
        dw = tr x <> d
        db = fromList $ sumElements `map` toColumns d
