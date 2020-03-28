{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE QuasiQuotes      #-}
{-# LANGUAGE TypeFamilies     #-}

module Learn
  ( newManager
  , defaultManagerSettings
  , predict
  , evaluate
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

learnForward ::
     (NElement e, ForwardLayer a e, OutputLayer b e)
  => ForwardNN a b
  -> TrainBatch e
  -> (BackwardNN (Backward a e) (Backput b e), e)
learnForward (ForwardNN layers loss) (TrainBatch (t, x)) =
  result `seq` (BackwardNN layers' loss', result)
  where
    (layers', y) = forward layers x
    (loss', result) = output loss t y

learnBackward ::
     (NElement e, BackwardLayer a' e, BackputLayer b' e)
  => e
  -> BackwardNN a' b'
  -> ForwardNN (Forward a') (Output b')
learnBackward rate (BackwardNN layers loss) = ForwardNN layers' loss'
  where
    (loss', d) = backput loss
    (layers', _) = backward rate layers d

learn ::
     ( NElement e
     , ForwardLayer a e
     , OutputLayer b e
     , BackwardLayer (Backward a e) e
     , BackputLayer (Backput b e) e
     )
  => e
  -> ForwardNN a b
  -> TrainBatch e
  -> (ForwardNN (Forward (Backward a e)) (Output (Backput b e)), e)
learn rate a = first (learnBackward rate) . learnForward a

learnAll rate origin batches = ls `seq` (nn, ls)
  where
    (nn, ls) = foldr f (origin, []) batches
    f batch (a, ls) = ls `seq` second (~: ls) $ learn rate a batch
    x ~: xs = trace [i|[#{length xs + 1}/#{n}] #{show x}|] (x : xs)
    n = length batches

predict :: (NElement e, ForwardLayer a e) => a -> Vector e -> Int
predict layers = maxIndex . flatten . snd . forward layers . asRow

evaluate :: (NElement e, ForwardLayer a e) => a -> [(Int, Vector e)] -> Double
evaluate layers samples = fromIntegral nOk / fromIntegral (length samples)
  where
    nOk = length $ filter (uncurry (==)) results
    results = map (second $ predict layers) samples

normalAffine nIn nOut = do
  weights <- rand nIn nOut
  bias <- flatten <$> rand nOut 1
  return $ AffineForward weights bias

initNN ::
     (NElement e, ForwardLayer x e, ForwardLayer a e, OutputLayer b e)
  => x
  -> [Int]
  -> IO (ForwardNN a b e)
initNN eoa ns = do
  (lastAffine:affines) <- mapM (uncurry normalAffine) $ spans [] ns
  let layers = foldl' (\b a -> a ~> eoa ~> b) lastAffine affines
  return $ ForwardNN layers SoftmaxWithCrossForward
  where
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
     ( Output (Backput b R) ~ b
     , Forward (Backward a R) ~ a
     , ForwardLayer a R
     , OutputLayer b R
     , BackwardLayer (Backward a R) R
     , BackputLayer (Backput b R) R
     )
  => R
  -> Int
  -> Int
  -> ForwardNN a b
  -> MnistData
  -> MnistData
  -> ([R], Double)
trainingSimple rate batchSize nReplicate nn trainData testData =
  (losses, evaluate layers $ convertTests testData)
  where
    trainBatches = convertTrains batchSize trainData
    batches = concat $ replicate nReplicate trainBatches
    (ForwardNN layers _, losses) = learnAll rate nn batches
