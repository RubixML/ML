<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use MathPHP\Statistics\Significance;
use RuntimeException;

/**
 * Z Scale Standardizer
 *
 * A way of centering and scaling a sample matrix by computing the Z Score for =
 * each feature.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ZScaleStandardizer implements Transformer
{
    /**
     * The computed means of the fitted data indexed by column.
     *
     * @var array|null
     */
    protected $means;

    /**
     * The computed standard deviations of the fitted data indexed by column.
     *
     * @var array|null
     */
    protected $stddevs;

    /**
     * Return the means calculated by fitting the training set.
     *
     * @return  array
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * Return the standard deviations calculated during fitting.
     *
     * @return  array
     */
    public function stddevs() : array
    {
        return $this->stddevs;
    }

    /**
     * Calculate the means and standard deviations of the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->means = $this->stddevs = [];

        foreach ($dataset->rotate() as $column => $values) {
            if ($dataset->type($column) === self::CONTINUOUS) {
                $mean = Average::mean($values);

                $deviations = [];

                foreach ($values as $value) {
                    $deviations[] = ($value - $mean) ** 2;
                }

                $this->stddevs[$column] = sqrt(Average::mean($deviations))
                    + self::EPSILON;

                $this->means[$column] = $mean;
            }
        }
    }

    /**
     * Transform the features into a z score.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (!isset($this->means) or !isset($this->stddevs)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($sample as $column => &$feature) {
                $feature = Significance::zScore($feature,
                    $this->means[$column], $this->stddevs[$column]);
            }
        }
    }
}
