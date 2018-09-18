<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Structures\DataFrame;
use RuntimeException;

/**
 * Z Scale Standardizer
 *
 * A way of centering and scaling a sample matrix by computing the Z Score for
 * each feature.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ZScaleStandardizer implements Transformer
{
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
     * Return the means calculated by fitting the training set.
     *
     * @return array|null
     */
    public function means() : ?array
    {
        return $this->means;
    }

    /**
     * Return the standard deviations calculated during fitting.
     *
     * @return array|null
     */
    public function stddevs() : ?array
    {
        return $this->stddevs;
    }

    /**
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Other\Structures\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        $this->means = $this->stddevs = [];

        foreach ($dataframe->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                list($mean, $variance) = Stats::meanVar($dataframe->column($column));

                $this->means[$column] = $mean;
                $this->stddevs[$column] = sqrt($variance);
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
        if (is_null($this->means) or is_null($this->stddevs)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->means as $column => $mean) {
                $feature = $sample[$column];

                if ($this->center === true) {
                    $feature -= $mean;
                }

                if ($this->scale === true) {
                    $stddev = $this->stddevs[$column];

                    $feature = $stddev !== 0. ? $feature / $stddev : 1.;
                }

                $sample[$column] = $feature;
            }
        }
    }
}
