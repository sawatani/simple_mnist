{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE QuasiQuotes      #-}

module Learn
  ( newManager
  , defaultManagerSettings
  , predict
  , evaluate
  , genAffine
  , genBatchNorm
  , initNN
  , hotone
  , convertTrains
  , convertTests
  , trainingSimple
  , chunksOf
  ) where

import           Data.Bifunctor
import           Data.List
import           Data.List.Split
import           Data.String.Interpolate as S (i)
import           Debug.Trace
import           Layers
import           Mnist
import           Network.HTTP.Client
import           Numeric.LinearAlgebra
import           Synapse

learnForward :: NElement a => ForwardNN a -> TrainBatch a -> (BackwardNN a, a)
learnForward (ForwardNN layers loss) (TrainBatch (t, x)) =
  result `seq` (BackwardNN layers' loss', result)
  where
    (layers', y) = forward layers x
    (loss', result) = output loss t y

learnBackward :: NElement a => a -> BackwardNN a -> ForwardNN a
learnBackward rate (BackwardNN layers loss) = ForwardNN layers' loss'
  where
    (loss', d) = backput loss
    (layers', _) = backward rate layers d

learn :: NElement a => a -> ForwardNN a -> TrainBatch a -> (ForwardNN a, a)
learn rate a = first (learnBackward rate) . learnForward a

learnAll ::
     NElement a => a -> ForwardNN a -> [TrainBatch a] -> (ForwardNN a, [a])
learnAll rate origin batches = ls `seq` (nn, ls)
  where
    (nn, ls) = foldr f (origin, []) batches
    f batch (a, ls) = second (~: ls) $ learn rate a batch
    x ~: xs =
      let r = (x : xs)
       in trace [i|[#{length r}/#{n}] #{show x}|] r
    n = length batches

predict :: NElement a => ForwardLayer a -> Vector a -> Int
predict layers = maxIndex . flatten . snd . forward layers . asRow

evaluate :: NElement a => ForwardLayer a -> [(Int, Vector a)] -> Double
evaluate layers samples = fromIntegral nOk / fromIntegral (length samples)
  where
    nOk = length $ filter (uncurry (==)) results
    results = map (second $ predict layers) samples

genAffine :: Int -> Int -> IO (ForwardLayer R)
genAffine nIn nOut = do
  weights <- rand nIn nOut
  bias <- flatten <$> rand nOut 1
  return $ AffineForward weights bias

genBatchNorm :: Int -> IO (ForwardLayer R)
genBatchNorm n = do
  rows <- rand 2 n
  let [gamma, beta] = toRows rows
  return $ BatchNormForward (BatchNormParam gamma beta)

initNN :: (Int -> IO (ForwardLayer R)) -> [Int] -> IO (ForwardNN R)
initNN genMid ns = do
  let ((x, y):ps) = spans [] ns
  layers <- foldl' join (genAffine x y) ps
  return $ ForwardNN layers SoftmaxWithCrossForward
  where
    join pre (x, y) = do
      b <- pre
      o <- genMid y
      a <- genAffine x y
      return $ a ~> o ~> b
    spans rs [a, b]   = (a, b) : rs
    spans rs (a:b:xs) = spans ((a, b) : rs) (b : xs)

convertTrains :: Int -> MnistData -> [TrainBatch R]
convertTrains batchSize (MnistData src) =
  map (mkTrainer . unzip) $ chunksOf batchSize vectors
  where
    vectors = map (bimap (hotone 10) flatten) src
    mkTrainer (a, b) = TrainBatch (fromRows a, fromZ (fromRows b) / 255)

hotone :: (Integral v, NElement a) => Int -> v -> Vector a
hotone n' v = fromList $ map fromIntegral list
  where
    n =
      if i < n'
        then n'
        else error ("Too large value: " ++ show i)
    list = replicate i 0 ++ 1 : replicate (n - i - 1) 0
    i = fromIntegral v

convertTests :: MnistData -> [(Int, Vector R)]
convertTests (MnistData src) =
  map (bimap fromIntegral $ (/ 255) . flatten . fromZ) src

trainingSimple ::
     Double
  -> Int
  -> Int
  -> ForwardNN R
  -> MnistData
  -> MnistData
  -> ([Double], Double)
trainingSimple rate batchSize nReplicate nn trainData testData =
  (losses, evaluate layers $ convertTests testData)
  where
    trainBatches = convertTrains batchSize trainData
    batches = concat $ replicate nReplicate trainBatches
    (ForwardNN layers _, losses) = learnAll rate nn batches
