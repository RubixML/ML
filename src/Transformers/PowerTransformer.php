<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function is_finite;
use function abs;
use function log;
use function exp;
use function array_values;
use function array_filter;

use const Rubix\ML\EPSILON;

/**
 * Power Transformer
 *
 * A family of parametric, monotonic transformations that apply the Yeo-Johnson transformation to the
 * features of a dataset in order to make their distributions more Gaussian-like. The transformation
 * parameter (lambda) is estimated per feature via maximum likelihood. Unlike the Box-Cox family, the
 * Yeo-Johnson family is defined for negative and zero values.
 *
 * $$
 * {\displaystyle z = \begin{cases} { (x + 1)^\lambda - 1 \over \lambda } & x \ge 0, \lambda \neq 0 \\ \ln{(x + 1)} & x \ge 0, \lambda = 0 \\ -{ ((1 - x)^{2 - \lambda} - 1) \over 2 - \lambda } & x < 0, \lambda \neq 2 \\ -\ln{(1 - x)} & x < 0, \lambda = 2 \end{cases}}
 * $$
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PowerTransformer implements Transformer, Stateful, Reversible, Persistable
{
    use AutotrackRevisions;

    /**
     * The constant of the golden ratio conjugate.
     *
     * @var float
     */
    protected const float GOLDEN_RATIO = 0.618033988749895;

    /**
     * The lower bound of the lambda search range.
     *
     * @var float
     */
    protected const float LAMBDA_MIN = -5.0;

    /**
     * The upper bound of the lambda search range.
     *
     * @var float
     */
    protected const float LAMBDA_MAX = 5.0;

    /**
     * A user-specified lambda to apply to all features. When null, the lambda is estimated per feature
     * via maximum likelihood during fitting.
     *
     * @var float|null
     */
    protected ?float $lambda = null;

    /**
     * The estimated transformation parameters indexed by column.
     *
     * @var float[]|null
     */
    protected ?array $lambdas = null;

    /**
     * Apply the Yeo-Johnson transformation to a single value.
     *
     * @param float $value
     * @param float $lambda
     * @return float
     */
    protected static function yeoJohnson(float $value, float $lambda) : float
    {
        if ($value >= 0.0) {
            if (abs($lambda) > EPSILON) {
                return ((($value + 1.0) ** $lambda) - 1.0) / $lambda;
            }

            return log($value + 1.0);
        }

        $gamma = 2.0 - $lambda;

        if (abs($gamma) > EPSILON) {
            return -((((1.0 - $value) ** $gamma) - 1.0) / $gamma);
        }

        return -log(1.0 - $value);
    }

    /**
     * Apply the inverse of the Yeo-Johnson transformation to a single value.
     *
     * @param float $value
     * @param float $lambda
     * @return float
     */
    protected static function inverseYeoJohnson(float $value, float $lambda) : float
    {
        if ($value >= 0.0) {
            if (abs($lambda) > EPSILON) {
                return ((($lambda * $value) + 1.0) ** (1.0 / $lambda)) - 1.0;
            }

            return exp($value) - 1.0;
        }

        $gamma = 2.0 - $lambda;

        if (abs($gamma) > EPSILON) {
            return 1.0 - ((1.0 - ($gamma * $value)) ** (1.0 / $gamma));
        }

        return 1.0 - exp(-$value);
    }

    /**
     * @param float|null $lambda
     * @throws InvalidArgumentException
     */
    public function __construct(?float $lambda = null)
    {
        if ($lambda !== null and !is_finite($lambda)) {
            throw new InvalidArgumentException('Lambda must be finite.');
        }

        $this->lambda = $lambda;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<DataType>
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
        return isset($this->lambdas);
    }

    /**
     * Return the estimated transformation parameters indexed by column.
     *
     * @return float[]|null
     */
    public function lambdas() : ?array
    {
        return $this->lambdas;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->lambdas = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isContinuous()) {
                if ($this->lambda !== null) {
                    $this->lambdas[$column] = $this->lambda;
                } else {
                    $values = array_values(array_filter($dataset->feature($column), 'is_finite'));

                    if (count($values) < 2 or Stats::variance($values) < EPSILON) {
                        $this->lambdas[$column] = 1.0;
                    } else {
                        $this->lambdas[$column] = $this->estimateLambda($values);
                    }
                }
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->lambdas === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->lambdas as $column => $lambda) {
                $value = &$sample[$column];

                if (!is_finite($value)) {
                    continue;
                }

                $value = self::yeoJohnson($value, $lambda);
            }
        }

        unset($sample);
    }

    /**
     * Perform the reverse transformation to the samples.
     *
     * @param list<list<mixed>> $samples
     * @throws RuntimeException
     */
    public function reverseTransform(array &$samples) : void
    {
        if ($this->lambdas === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->lambdas as $column => $lambda) {
                $value = &$sample[$column];

                if (!is_finite($value)) {
                    continue;
                }

                $value = self::inverseYeoJohnson($value, $lambda);
            }
        }

        unset($sample);
    }

    /**
     * Estimate the lambda that maximizes the profile log likelihood of the given values.
     *
     * @param list<int|float> $values
     * @return float
     */
    protected function estimateLambda(array $values) : float
    {
        $lo = self::LAMBDA_MIN;
        $ho = self::LAMBDA_MAX;

        $c = $ho - ($ho - $lo) * self::GOLDEN_RATIO;
        $d = $lo + ($ho - $lo) * self::GOLDEN_RATIO;

        $fc = $this->logLikelihood($c, $values);
        $fd = $this->logLikelihood($d, $values);

        for ($i = 0; $i < 100; ++$i) {
            if ($fc >= $fd) {
                $ho = $d;
                $d = $c;
                $fd = $fc;

                $c = $ho - ($ho - $lo) * self::GOLDEN_RATIO;
                $fc = $this->logLikelihood($c, $values);
            } else {
                $lo = $c;
                $c = $d;
                $fc = $fd;

                $d = $lo + ($ho - $lo) * self::GOLDEN_RATIO;
                $fd = $this->logLikelihood($d, $values);
            }
        }

        return ($lo + $ho) / 2.0;
    }

    /**
     * Compute the profile log likelihood of a set of transformed values under a given lambda.
     *
     * @param float $lambda
     * @param list<int|float> $values
     * @return float
     */
    protected function logLikelihood(float $lambda, array $values) : float
    {
        $n = count($values);

        $transformed = [];
        $sigma = 0.0;

        foreach ($values as $value) {
            $transformed[] = self::yeoJohnson($value, $lambda);

            $sigma += $value >= 0.0 ? log($value + 1.0) : -log(1.0 - $value);
        }

        $variance = Stats::variance($transformed);

        if ($variance <= 0.0 or !is_finite($variance)) {
            return -INF;
        }

        return -($n / 2.0) * log($variance) + ($lambda - 1.0) * $sigma;
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
        return 'Power Transformer (lambda: ' . Params::toString($this->lambda) . ')';
    }
}
