<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
use RuntimeException;

/**
 * Quartile Standardizer
 *
 * This standardizer removes the median and scales each sample according to the
 * interquantile range (IQR). The IQR is the range between the 1st quartile
 * (25th quantile) and the 3rd quartile (75th quantile).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class QuartileStandardizer implements Transformer, Stateful
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
     * The computed interquartile ranges of the fitted data indexed by column.
     *
     * @var array|null
     */
    protected $iqrs;

    /**
     * @param  bool  $center
     * @return void
     */
    public function __construct(bool $center = true)
    {
        $this->center = $center;
    }

    /**
     * Return the means calculated by fitting the training set.
     *
     * @return array|null
     */
    public function medians() : ?array
    {
        return $this->medians;
    }

    /**
     * Return the interquartile ranges calculated during fitting.
     *
     * @return array|null
     */
    public function iqrs() : ?array
    {
        return $this->iqrs;
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->medians = $this->iqrs = [];

        foreach ($dataset->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                $values = $dataset->column($column);

                $median = Stats::median($values);
                $iqr = Stats::iqr($values);

                $this->medians[$column] = $median;
                $this->iqrs[$column] = $iqr ?: self::EPSILON;
            }
        }
    }

    /**
     * Apply the transformation to the samples in the data frame.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->medians) or is_null($this->iqrs)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->iqrs as $column => $iqr) {
                $feature = $sample[$column];

                if ($this->center === true) {
                    $feature -= $this->medians[$column];
                }

                $sample[$column] = $feature / $iqr;
            }
        }
    }
}
