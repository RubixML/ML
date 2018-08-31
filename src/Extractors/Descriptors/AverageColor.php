<?php

namespace Rubix\ML\Extractors\Descriptors;

use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

/**
 * Average Color
 *
 * This descriptor computes the average color intensity for each of the 3 color
 * channels (red, green, and blue).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AverageColor implements Descriptor
{
    /**
     * Extract features from an image patch and return them in an array.
     *
     * @param  array  $patch
     * @return array
     */
    public function describe(array $patch) : array
    {
        $n = count($patch) + count(reset($patch));

        $r = $g = $b = 0;

        foreach ($patch as $pixels) {
            foreach ($pixels as $rgb) {
                $r += $rgb[0];
                $g += $rgb[1];
                $b += $rgb[2];
            }
        }

        $r = (int) round($r / $n);
        $g = (int) round($g / $n);
        $b = (int) round($b / $n);

        return [$r, $g, $b];
    }
}
