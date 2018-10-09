<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
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
class ZScaleStandardizer implements Transformer, Online
{
    /**
     * Should we center the data?
     *
     * @var bool
     */
    protected $center;

    /**
     * The means of each feature column from the fitted data.
     *
     * @var array|null
     */
    protected $means;

    /**
     * The variances of each feature column from the fitted data.
     *
     * @var array|null
     */
    protected $variances;

    /**
     *  The number of samples that this tranformer has fitted.
     * 
     * @var int|null
     */
    protected $n;

    /**
     * The precomputed standard deviations.
     *
     * @var array|null
     */
    protected $stddevs;

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
    public function means() : ?array
    {
        return $this->means;
    }

    /**
     * Return the variances calculated by fitting the training set.
     *
     * @return array|null
     */
    public function variances() : ?array
    {
        return $this->variances;
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
     * Fit the transformer to the data.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        $this->means = $this->variances = $this->stddevs = [];

        foreach ($dataframe->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                $values = $dataframe->column($column);

                list($mean, $variance) = Stats::meanVar($values);

                $this->means[$column] = $mean;
                $this->variances[$column] = $variance;
                $this->stddevs[$column] = sqrt($variance ?: self::EPSILON);
            }
        }

        $this->n = $dataframe->numRows();
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function update(DataFrame $dataframe) : void
    {
        if (is_null($this->means) or is_null($this->variances)) {
            $this->fit($dataframe);
            return;
        }

        $n = $dataframe->numRows();

        foreach ($this->means as $column => $oldMean) {
            $oldVariance = $this->variances[$column];

            $values = $dataframe->column($column);

            list($mean, $variance) = Stats::meanVar($values);

            $this->means[$column] = (($n * $mean)
                + ($this->n * $oldMean))
                / ($this->n + $n);

            $temp = ($this->n
                * $oldVariance + ($n * $variance)
                + ($this->n / ($n * ($this->n + $n)))
                * ($n * $oldMean - $n * $mean) ** 2)
                / ($this->n + $n);

            $this->variances[$column] = $temp;
            $this->stddevs[$column] = sqrt($temp ?: self::EPSILON);
        }

        $this->n += $n;
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
        if (is_null($this->means) or is_null($this->stddevs)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->stddevs as $column => $stddev) {
                $feature = $sample[$column];

                if ($this->center === true) {
                    $feature -= $this->means[$column];
                }

                $sample[$column] = $feature / $stddev;
            }
        }
    }
}
