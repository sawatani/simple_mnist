module Mnist
    (
      MnistData(..)
    , MnistLabels(..)
    , MnistImages(..)
    , timestamp
    , saveMnist
    , readMnist
    , zipMnist
    ) where

import Synapse
import Layers
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
import           Prelude                     hiding ((<>))
import           System.Directory
import           System.FilePath.Posix
import           System.IO
import           System.Random.MWC

newtype MnistLabels = MnistLabels [Word8] deriving (Show)
newtype MnistImages = MnistImages [Matrix Z] deriving (Show)
newtype MnistData = MnistData [(Word8, Matrix Z)] deriving (Show)

zipMnist :: MnistLabels -> MnistImages -> MnistData
zipMnist (MnistLabels labels) (MnistImages images) = list `deepseq` MnistData list
    where list = labels `zip` images

markMinistLabels = 2049 :: Int
markMnistImages = 2051 :: Int

putAsWord32 :: Integral a => a -> Put
putAsWord32 a = B.put (fromIntegral a :: Word32)

getAsIntegral :: Integral a => Get a
getAsIntegral = fromIntegral <$> (B.get :: Get Word32)

instance B.Binary MnistLabels where
    put (MnistLabels labels) = do
        B.put markMinistLabels
        putAsWord32 $ length labels
        B.put $ BS.pack labels

    get = do
        mark <- getAsIntegral
        guard $ mark == markMinistLabels
        size <- getAsIntegral
        let total = trace ("Reading labels: " ++ show size) size
        labels <- BGI.readN total BSS.unpack
        return $ MnistLabels $ labels

instance B.Binary MnistImages where
    put (MnistImages (images @ (hm:_))) = do
        putAsWord32 markMnistImages
        putAsWord32 $ length images
        putAsWord32 $ rows hm
        putAsWord32 $ cols hm
        B.put $ (BS.pack . map fromIntegral . concat . concat . map toLists) images

    get = do
        mark <- getAsIntegral
        guard $ mark == markMnistImages
        size <- getAsIntegral
        nRows <- getAsIntegral
        nCols <- getAsIntegral
        let total = trace ("Reading images: " ++ show size ++ " of " ++ show nRows ++ "x" ++ show nCols) size * nRows * nCols
        zs <- map fromIntegral <$> BGI.readN total BSS.unpack
        let images = map (nRows >< nCols) $ chunksOf (nRows * nCols) zs
        return $ MnistImages images


timestamp :: String -> IO ()
timestamp msg = msg `deepseq` do
    t <- getZonedTime
    let fm = "[%Y/%m/%d %H:%M:%S] " ++ msg
    print $ formatTime defaultTimeLocale fm t

download :: Manager -> String -> IO BS.ByteString
download manager url = do
   request <- parseRequest url
   timestamp $ "Downloading " ++ url
   response <- httpLbs request manager
   return $ responseBody response

saveMnist :: FilePath -> String -> [String] -> Manager -> IO [FilePath]
saveMnist rootDir urlBase filenames manager = do
    isDir <- doesDirectoryExist rootDir
    _ <- if isDir then return () else createDirectory rootDir
    mapM saveFile filenames
    where
        saveFile filename = do
            let url = urlBase ++ filename
            let file = rootDir </> filename
            timestamp $ "Checking file " ++ file
            e <- doesFileExist file
            if e then return () else download manager url >>= (BS.writeFile file)
            return file

readMnist :: FilePath -> IO (Either MnistLabels MnistImages)
readMnist file = do
    timestamp $ "decoding " ++ file ++ " ..."
    bs <- GZip.decompress <$> BS.readFile file
    let a = left (const $ B.decode bs) $ right (\(_, _, a) -> a) $ B.decodeOrFail bs
    timestamp $ "decoded " ++ file
    return a
