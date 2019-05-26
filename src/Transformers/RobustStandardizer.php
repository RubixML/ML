<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use RuntimeException;

use const Rubix\ML\EPSILON;

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
class RobustStandardizer implements Transformer, Stateful
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
     * The computed median absolute deviations of the fitted data
     * indexed by column.
     *
     * @var array|null
     */
    protected $mads;

    /**
     * @param bool $center
     */
    public function __construct(bool $center = true)
    {
        $this->center = $center;
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return $this->medians and $this->mads;
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
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        $n = $dataset->numColumns();

        $this->medians = $this->mads = [];

        for ($column = 0; $column < $n; $column++) {
            if ($dataset->columnType($column) === DataType::CONTINUOUS) {
                $values = $dataset->column($column);
                
                [$median, $mad] = Stats::medianMad($values);

                $this->medians[$column] = $median;
                $this->mads[$column] = $mad ?: EPSILON;
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->medians === null or $this->mads === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->mads as $column => $mad) {
                $feature = $sample[$column];

                if ($this->center) {
                    $feature -= $this->medians[$column];
                }

                $sample[$column] = $feature / $mad;
            }
        }
    }
}
