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

data ForwardLayer a
  = AffineForward (Weight a) (Bias a)
  | SigmoidForward
  | ReLUForward
  | JoinedForwardLayer (ForwardLayer a) (ForwardLayer a)
  deriving (Show)

instance (Numeric a, Eq a) => Eq (ForwardLayer a) where
  AffineForward w b == AffineForward w' b' = w == w' && b == b'
  SigmoidForward == SigmoidForward = True
  ReLUForward == ReLUForward = True
  JoinedForwardLayer a b == JoinedForwardLayer a' b' = a == a' && b == b'
  _ == _ = False

infixl 4 ~>

(~>) :: ForwardLayer a -> ForwardLayer a -> ForwardLayer a
a ~> (JoinedForwardLayer x y) = (a ~> x) ~> y
a ~> b = JoinedForwardLayer a b

data OutputLayer a =
  SoftmaxWithCrossForward
  deriving (Show)

data ForwardNN a =
  ForwardNN (ForwardLayer a) (OutputLayer a)
  deriving (Show)

data BackwardLayer a
  = AffineBackward (Weight a) (Bias a) (SignalX a)
  | SigmoidBackward (SignalY a)
  | ReLUBackward (SignalX a)
  | JoinedBackwardLayer (BackwardLayer a) (BackwardLayer a)
  deriving (Show)

instance (Numeric a, Eq a) => Eq (BackwardLayer a) where
  AffineBackward w b d == AffineBackward w' b' d' =
    w == w' && b == b' && d == d'
  SigmoidBackward y == SigmoidBackward y' = y == y'
  ReLUBackward x == ReLUBackward x' = x == x'
  JoinedBackwardLayer a b == JoinedBackwardLayer a' b' = a == a' && b == b'
  _ == _ = False

infixr 4 <~

(<~) :: BackwardLayer a -> BackwardLayer a -> BackwardLayer a
(JoinedBackwardLayer x y) <~ b = x <~ (y <~ b)
a <~ b = JoinedBackwardLayer a b

data BackputLayer a =
  SoftmaxWithCrossBackward (TeacherBatch a) (SignalY a)

data BackwardNN a =
  BackwardNN (BackwardLayer a) (BackputLayer a)

type NElement a = (Ord a, Floating a, Numeric a, Num (Vector a), Show a)

forward ::
     NElement a => ForwardLayer a -> SignalX a -> (BackwardLayer a, SignalY a)
forward (AffineForward w b) x = (AffineBackward w b x, affinem w b x)
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
