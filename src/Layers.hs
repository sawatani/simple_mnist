{-# LANGUAGE ConstraintKinds  #-}
{-# LANGUAGE FlexibleContexts #-}

module Layers
  ( ForwardLayer(..)
  , OutputLayer(..)
  , ForwardNN(..)
  , BackwardLayer(..)
  , BackputLayer(..)
  , BackwardNN(..)
  , TrainBatch(..)
  , NElement(..)
  , (<~)
  , (~>)
  , forward
  , output
  , backward
  , backput
  ) where

import           Control.Arrow
import           Control.Lens          hiding ((<~))
import           Debug.Trace
import           Numeric
import           Numeric.LinearAlgebra
import           Synapse

type Weight a = Matrix a

type Bias a = Vector a

type SignalX a = Matrix a

type SignalY a = Matrix a

type Diff a = Matrix a

type TeacherBatch a = Matrix a

type InputBatch a = Matrix a

newtype TrainBatch a =
  TrainBatch (TeacherBatch a, InputBatch a)
  deriving (Show)

data Numeric a =>
     ForwardLayer a
  = AffineForward (Weight a) (Bias a)
  | BatchNormForward (BatchNormParam a)
  | SigmoidForward
  | ReLUForward
  | JoinedForwardLayer (ForwardLayer a) (ForwardLayer a)
  deriving (Show, Eq)

infixr 4 ~>

(~>) :: Numeric a => ForwardLayer a -> ForwardLayer a -> ForwardLayer a
(JoinedForwardLayer x y) ~> a = x ~> y ~> a
a ~> b = JoinedForwardLayer a b

data OutputLayer a =
  SoftmaxWithCrossForward
  deriving (Show, Eq)

data ForwardNN a =
  ForwardNN (ForwardLayer a) (OutputLayer a)
  deriving (Show, Eq)

data Numeric a =>
     BackwardLayer a
  = AffineBackward (Weight a) (Bias a) (SignalX a)
  | BatchNormBackward (BatchNormParam a) (BatchNormCache a)
  | SigmoidBackward (SignalY a)
  | ReLUBackward (SignalX a)
  | JoinedBackwardLayer (BackwardLayer a) (BackwardLayer a)
  deriving (Show, Eq)

infixl 4 <~

(<~) :: Numeric a => BackwardLayer a -> BackwardLayer a -> BackwardLayer a
b <~ (JoinedBackwardLayer x y) = b <~ x <~ y
a <~ b = JoinedBackwardLayer a b

data Numeric a =>
     BackputLayer a =
  SoftmaxWithCrossBackward (TeacherBatch a) (SignalY a)
  deriving (Show, Eq)

data BackwardNN a =
  BackwardNN (BackwardLayer a) (BackputLayer a)
  deriving (Show, Eq)

type NElement a
   = (Ord a, Floating a, Numeric a, Floating (Vector a), Num (Vector a), Show a)

forward ::
     NElement a => ForwardLayer a -> SignalX a -> (BackwardLayer a, SignalY a)
forward (AffineForward w b) x = (AffineBackward w b x, affinem w b x)
forward (BatchNormForward param) x = (BatchNormBackward param cache, out)
  where
    (cache, out) = batchNormm param x
forward SigmoidForward x = (SigmoidBackward y, y)
  where
    y = sigmoidm x
forward ReLUForward x = (ReLUBackward x, relum x)
forward (JoinedForwardLayer a b) x0 = (a' <~ b', x2)
  where
    (a', x1) = forward a x0
    (b', x2) = forward b x1

backward ::
     NElement a => a -> BackwardLayer a -> Diff a -> (ForwardLayer a, Diff a)
backward rate (AffineBackward w b x) d =
  (AffineForward (w - scale rate w') (b - scale rate b'), x')
  where
    (x', w', b') = affinemBackward w x d
backward rate (BatchNormBackward (BatchNormParam gamma beta) cache) d =
  (BatchNormForward param, d')
  where
    param = BatchNormParam (gamma - scale rate dgamma) (beta - scale rate dbeta)
    (dgamma, dbeta, d') = batchNormmBackward cache d
backward _ (SigmoidBackward y) d = (SigmoidForward, sigmoidBackward y d)
backward _ (ReLUBackward x) d = (ReLUForward, relumBackward x d)
backward r (JoinedBackwardLayer a b) d0 = (a' ~> b', d2)
  where
    (b', d1) = backward r b d0
    (a', d2) = backward r a d1

output ::
     NElement a
  => OutputLayer a
  -> TeacherBatch a
  -> SignalY a
  -> (BackputLayer a, a)
output SoftmaxWithCrossForward t y = (SoftmaxWithCrossBackward t y', loss)
  where
    (y', loss) = softmaxWithCross t y

backput :: NElement a => BackputLayer a -> (OutputLayer a, Diff a)
backput (SoftmaxWithCrossBackward t y) =
  (SoftmaxWithCrossForward, softmaxWithCrossBackward t y)
