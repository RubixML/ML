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
class ZScaleStandardizer implements Transformer, Stateful, Elastic, Reversible, Persistable
{
    use AutotrackRevisions;

    /**
     * Should we center the data at 0?
     *
     * @var bool
     */
    protected bool $center;

    /**
     * The means of each continuous feature column of the fitted data.
     *
     * @var float[]|null
     */
    protected ?array $means = null;

    /**
     * The variances of each continuous feature column of the fitted data.
     *
     * @var float[]|null
     */
    protected ?array $variances = null;

    /**
     *  The number of samples that this transformer has fitted.
     *
     * @var int
     */
    protected int $n = 0;

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
     * Return the means of the fitted continuous features.
     *
     * @return float[]|null
     */
    public function means() : ?array
    {
        return $this->means;
    }

    /**
     * Return the variances of the fitted continuous features.
     *
     * @return float[]|null
     */
    public function variances() : ?array
    {
        return $this->variances;
    }

    /**
     * Return the standard deviations of the fitted continuous features.
     *
     * @return float[]|null
     */
    public function stddevs() : ?array
    {
        return isset($this->variances) ? array_map('sqrt', $this->variances) : null;
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

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $values = $dataset->feature($column);

                [$mean, $variance] = Stats::meanVar($values);

                $this->means[$column] = $mean;
                $this->variances[$column] = $variance;
            }
        }

        $this->n = $dataset->numSamples();
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

        $n = $dataset->numSamples();

        foreach ($this->means as $column => $oldMean) {
            $oldVariance = $this->variances[$column];

            $values = $dataset->feature($column);

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
        if ($this->means === null or $this->variances === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->variances as $column => $variance) {
                $value = &$sample[$column];

                if ($this->center) {
                    $value -= $this->means[$column];
                }

                if ($variance > 0.0) {
                    $value /= sqrt($variance);
                }
            }
        }
    }

    /**
     * Perform the reverse transformation to the samples.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function reverseTransform(array &$samples) : void
    {
        if ($this->means === null or $this->variances === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->variances as $column => $variance) {
                $value = &$sample[$column];

                if ($variance > 0.0) {
                    $value *= sqrt($variance);
                }

                if ($this->center) {
                    $value += $this->means[$column];
                }
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Z Scale Standardizer (center: ' . Params::toString($this->center) . ')';
    }
}
