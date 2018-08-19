<?php

namespace Rubix\ML\Extractors\Descriptors;

use MathPHP\Statistics\Average;
use Rubix\ML\Other\Functions\MeanVar;
use MathPHP\Statistics\RandomVariable;
use InvalidArgumentException;

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

        foreach ($patch as $x => $pixels) {
            foreach ($pixels as $y => $rgb) {
                $intensities[] = Average::mean($rgb);
            }
        }

        list($mean, $variance) = MeanVar::compute($intensities);

        $skewness = RandomVariable::skewness($intensities);

        $kurtosis = RandomVariable::kurtosis($intensities);

        return [$mean, $variance, $skewness, $kurtosis];
    }
}
