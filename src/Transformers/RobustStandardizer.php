<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Structures\DataFrame;
use RuntimeException;

/**
 * Robust Standardizer
 *
 * This Transformer standardizes continuous features by removing the median and
 * dividing over the median absolute deviation (MAD), a value referred to as
 * robust z score. The use of robust statistics makes this standardizer more
 * immune to outliers than the Z Scale Standardizer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RobustStandardizer implements Transformer
{
    const LAMBDA = 0.6745;

    /**
     * Should we center the data?
     *
     * @var bool
     */
    protected $center;

    /**
     * Should we scale the data?
     *
     * @var bool
     */
    protected $scale;

    /**
     * The computed medians of the fitted data indexed by column.
     *
     * @var array|null
     */
    protected $medians;

    /**
     * The computed median absolute deviations of the fitted data indexed by
     * column.
     *
     * @var array|null
     */
    protected $mads;

    /**
     * @param  bool  $center
     * @param  bool  $scale
     * @return void
     */
    public function __construct(bool $center = true, bool $scale = true)
    {
        $this->center = $center;
        $this->scale = $scale;
    }

    /**
     * Return the medians calculated by fitting the training set.
     *
     * @return array|null
     */
    public function medians() : ?array
    {
        return $this->medians;
    }

    /**
     * Return the median absolute deviations calculated during fitting.
     *
     * @return array|null
     */
    public function mads() : ?array
    {
        return $this->mads;
    }

    /**
     * Calculate the medians and median absolute deviations of the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->medians = $this->mads = [];

        foreach ($dataset->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                list($median, $mad) = Stats::medMad($dataset->column($column));

                $this->medians[$column] = $median;
                $this->mads[$column] = $mad;
            }
        }
    }

    /**
     * Center and scale the features of the sample matrix.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->medians) or is_null($this->mads)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->medians as $column => $median) {
                $feature = $sample[$column];

                if ($this->center === true) {
                    $feature -= $median;
                }

                if ($this->scale === true) {
                    $mad = $this->mads[$column];

                    $feature = $mad !== 0.
                        ? (self::LAMBDA * $feature) / $mad
                        : 1.;
                }

                $sample[$column] = $feature;
            }
        }
    }
}
