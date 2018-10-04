<?php

namespace Rubix\ML\Extractors\Descriptors;

use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

/**
 * Texture Histogram
 *
 * This descriptor computes the a histogram (represented as mean, variance,
 * skewness, and kurtosis) of color intensities of an image patch.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TextureHistogram implements Descriptor
{
    /**
     * Extract features from an image patch and return them in an array.
     *
     * @param  array  $patch
     * @return array
     */
    public function describe(array $patch) : array
    {
        $intensities = [];

        foreach ($patch as $pixels) {
            foreach ($pixels as $rgb) {
                $intensities[] = Stats::mean($rgb);
            }
        }

        list($mean, $variance) = Stats::meanVar($intensities);

        $skewness = Stats::skewness($intensities, $mean);

        $kurtosis = Stats::kurtosis($intensities, $mean);

        return [$mean, $variance, $skewness, $kurtosis];
    }
}
