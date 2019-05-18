<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Z Scale Standardizer
 *
 * A method of centering and scaling a dataset such that it has 0 mean and unit
 * variance (Z Score).
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ZScaleStandardizer implements Transformer, Stateful, Elastic
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
        return $this->means and $this->variances;
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
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        $columns = $dataset->columnsByType(DataType::CONTINUOUS);

        $this->means = $this->variances = $this->stddevs = [];

        foreach ($columns as $column => $values) {
            [$mean, $variance] = Stats::meanVar($values);

            $this->means[$column] = $mean;
            $this->variances[$column] = $variance;
            $this->stddevs[$column] = sqrt($variance ?: EPSILON);
        }

        $this->n = $dataset->numRows();
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function update(Dataset $dataset) : void
    {
        if ($this->means === null or $this->variances === null) {
            $this->fit($dataset);

            return;
        }

        $n = $dataset->numRows();

        foreach ($this->means as $column => $oldMean) {
            $oldVariance = $this->variances[$column];

            $values = $dataset->column($column);

            [$mean, $variance] = Stats::meanVar($values);

            $this->means[$column] = (($n * $mean)
                + ($this->n * $oldMean))
                / ($this->n + $n);

            $varNew = ($this->n
                * $oldVariance + ($n * $variance)
                + ($this->n / ($n * ($this->n + $n)))
                * ($n * $oldMean - $n * $mean) ** 2)
                / ($this->n + $n);

            $this->variances[$column] = $varNew;
            $this->stddevs[$column] = sqrt($varNew ?: EPSILON);
        }

        $this->n += $n;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->means === null or $this->stddevs === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->stddevs as $column => $stddev) {
                $feature = $sample[$column];

                if ($this->center) {
                    $feature -= $this->means[$column];
                }

                $sample[$column] = $feature / $stddev;
            }
        }
    }
}
