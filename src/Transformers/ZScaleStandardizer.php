<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\RuntimeException;

use function is_null;
use function sqrt;

/**
 * Z Scale Standardizer
 *
 * A method of centering and scaling a dataset such that it has 0 mean and unit variance, also known as a Z-Score. Although Z-Scores
 * are technically unbounded, in practice they mostly fall between -3 and 3 - that is, they are no more than 3 standard deviations
 * away from the mean.
 *
 * $$
 * {\displaystyle z = {x - \mu \over \sigma }}
 * $$
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ZScaleStandardizer implements Transformer, Stateful, Elastic, Persistable
{
    use AutotrackRevisions;

    /**
     * Should we center the data at 0?
     *
     * @var bool
     */
    protected $center;

    /**
     * The means of each feature column from the fitted data.
     *
     * @var float[]|null
     */
    protected $means;

    /**
     * The variances of each feature column from the fitted data.
     *
     * @var float[]|null
     */
    protected $variances;

    /**
     *  The number of samples that this transformer has fitted.
     *
     * @var int
     */
    protected $n = 0;

    /**
     * Return the inverse of the standard deviation while taking zeros and negatives into consideration.
     *
     * @param float $variance
     * @return float
     */
    protected static function stdInv(float $variance) : float
    {
        return $variance > 0.0 ? 1.0 / sqrt($variance) : 1.0;
    }

    /**
     * @param bool $center
     */
    public function __construct(bool $center = true)
    {
        $this->center = $center;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
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
     * @return float[]|null
     */
    public function means() : ?array
    {
        return $this->means;
    }

    /**
     * Return the variances calculated by fitting the training set.
     *
     * @return float[]|null
     */
    public function variances() : ?array
    {
        return $this->variances;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->means = $this->variances = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $values = $dataset->column($column);

                [$mean, $variance] = Stats::meanVar($values);

                $this->means[$column] = $mean;
                $this->variances[$column] = $variance;
            }
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

            $this->means[$column] = (($this->n * $oldMean)
                + ($n * $mean)) / ($this->n + $n);

            $this->variances[$column] = ($this->n
                * $oldVariance + ($n * $variance)
                + ($this->n / ($n * ($this->n + $n)))
                * ($n * $oldMean - $n * $mean) ** 2)
                / ($this->n + $n);
        }

        $this->n += $n;
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->means) or is_null($this->variances)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $stdInvs = array_map([self::class, 'stdInv'], $this->variances);

        foreach ($samples as &$sample) {
            foreach ($stdInvs as $column => $stdInv) {
                $value = &$sample[$column];

                if ($this->center) {
                    $value -= $this->means[$column];
                }

                $value *= $stdInv;
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Z Scale Standardizer (center: ' . Params::toString($this->center) . ')';
    }
}
