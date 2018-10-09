<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
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
    /**
     * Should we center the data?
     *
     * @var bool
     */
    protected $center;

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
     * @return void
     */
    public function __construct(bool $center = true)
    {
        $this->center = $center;
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
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        $this->medians = $this->mads = [];

        foreach ($dataframe->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                $values = $dataframe->column($column);
                
                list($median, $mad) = Stats::medMad($values);

                $this->medians[$column] = $median;
                $this->mads[$column] = $mad ?: self::EPSILON;
            }
        }
    }

    /**
     * Apply the transformation to the sample matrix.
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
            foreach ($this->mads as $column => $mad) {
                $feature = $sample[$column];

                if ($this->center === true) {
                    $feature -= $this->medians[$column];
                }

                $sample[$column] = $feature / $mad;
            }
        }
    }
}
