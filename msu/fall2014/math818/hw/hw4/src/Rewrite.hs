module Rewrite where

import Data.String.Utils
import Data.List
import Control.Monad
import Control.Arrow
import Math.NumberTheory.Factor

converge :: (Eq a) => (a -> a) -> a -> a
converge = until =<< ((==) =<<)

swapXY = replace "xy" "yyx"
swapYX = replace "yx" "xyyyy"
swapYZ = replace "yz" "zy"

alternating n xs = map test [1..n]
                   where test n = --map ((id &&& pfactors . toInteger) . length) .
                                  group .
                                  converge swapYX .
                                  concat .
                                  replicate n $ xs

main = mapM (putStrLn . show) $ alternating 2 "xy"
