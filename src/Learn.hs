module Learn
    (
      newManager
    , defaultManagerSettings
    , shuffleList
    , initNN
    , trainingSimple
    , chunksOf
    ) where

import Synapse
import Layers
import Mnist
import qualified Codec.Compression.GZip      as GZip
import           Control.Arrow
import           Control.DeepSeq
import           Control.Lens hiding ((<~))
import           Control.Monad               (join)
import           Control.Monad.ST
import           Control.Monad.State
import           Control.Monad.Trans.Class
import           Data.Binary                 as B
import           Data.Binary.Get.Internal    as BGI
import qualified Data.ByteString             as BSS
import qualified Data.ByteString.Lazy        as BS
import qualified Data.ByteString.Lazy.UTF8   as UTF8
import qualified Data.Conduit                as CD
-- import qualified Data.CSV.Conduit            as CSV
import           Data.Int
import           Data.List
import           Data.List.Split
import qualified Data.Map                    as Map
import           Data.Time.Clock
import           Data.Time.Format
import           Data.Time.LocalTime
import qualified Data.Vector                 as V
import qualified Data.Vector.Generic         as VG
import qualified Data.Vector.Generic.Mutable as VM
import           Debug.Trace
import           Network.HTTP.Client
import           Network.HTTP.Types.Status   (statusCode)
import           Numeric
import           Numeric.LinearAlgebra
import           Numeric.LinearAlgebra.Data
import           System.Directory
import           System.FilePath.Posix
import           System.IO
import           System.Random.MWC

normalAffine :: Int -> Int -> IO (ForwardLayer R)
normalAffine nIn nOut = do
        weights <- trace ("Random Affine: " ++ show nIn ++ "x" ++ show nOut) rand nIn nOut
        bias <- flatten <$> rand nOut 1
        return $ AffineForward weights bias

initNN :: ForwardLayer R -> [Int] -> IO (ForwardNN R)
initNN eoa ns = do
    (lastAffine:affines) <- mapM (uncurry normalAffine) $ spans [] ns
    let layers = foldl' (\b a -> a ~> eoa ~> b) lastAffine affines
    return $ ForwardNN layers SoftmaxWithCrossForward
        where
            spans rs (a:b:[]) = (a, b):rs
            spans rs (a:b:xs) = spans ((a, b):rs) (b:xs)

convertTrains :: Int -> MnistData -> [TrainBatch R]
convertTrains batchSize (MnistData src) = map (mkTrainer . unzip) $ chunksOf batchSize $ vectors
    where
        vectors = map (first (hotone 10) . second flatten) src
        mkTrainer (a, b) = TrainBatch (fromRows a, (fromZ $ fromRows b) / 255)

hotone :: (Integral v, NElement a) => Int -> v -> (Vector a)
hotone n v = fromList $ map fromIntegral list
    where
        list = replicate i 0 ++ 1 : replicate (n - i - 1) 0
        i = fromIntegral v

convertTests :: MnistData -> [(Int, Vector R)]
convertTests (MnistData src) = map (first fromIntegral . second ((/255) . flatten . fromZ)) src

trainingSimple :: Double -> Int -> Int -> ForwardNN R -> MnistData -> MnistData -> ([Double], Double)
trainingSimple rate batchSize nReplicate nn trainData testData = (losses, evaluate layers $ convertTests testData)
    where
        trainBatches = convertTrains batchSize trainData
        batches = concat $ replicate nReplicate trainBatches
        (ForwardNN layers _, losses) = learnAll rate nn batches


shuffleList :: [a] -> IO [a]
shuffleList [a] = return [a]
shuffleList xs =  withSystemRandom . asGenST $ \g -> do
    v <- VG.unsafeThaw $ VG.fromList xs
    repeatSwap v g n
    (v' :: V.Vector a) <- VG.unsafeFreeze v
    return $ VG.toList v'
    where
        n = length xs - 1
        repeatSwap v g i = if (i < 0) then return () else do
            j <- uniformR (0, n) g
            VM.swap v j i
            repeatSwap v g (i - 1)



-- saveCSV :: Show a => FilePath -> [String] -> [[a]] -> IO ()
-- saveCSV filePath header body = do
--     let rows = map (map show) body
--     CSV.writeCSVFile CSV.defCSVSettings filePath WriteMode $ header:rows
