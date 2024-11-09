<?php

namespace Rubix\ML\Datasets\Generators;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * Agglomerate
 *
 * An Agglomerate is a collection of other generators each assigned with a
 * user-definable label. Agglomerates are useful for classification,
 * clustering, and anomaly detection problems where the label is a discrete
 * value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Agglomerate implements Generator
{
    /**
     * An array of generators.
     *
     * @var \Rubix\ML\Datasets\Generators\Generator[]
     */
    protected array $generators;

    /**
     * The normalized weights of each generator i.e. the probability that a
     * sample from a particular generator shows up in the dataset.
     *
     * @var float[]
     */
    protected array $weights;

    /**
     * The dimensionality of the agglomerate.
     *
     * @var int
     */
    protected int $dimensions;

    /**
     * @param \Rubix\ML\Datasets\Generators\Generator[] $generators
     * @param (int|float)[]|null $weights
     * @throws InvalidArgumentException
     */
    public function __construct(array $generators = [], ?array $weights = null)
    {
        if (empty($generators)) {
            throw new InvalidArgumentException('Agglomerate must contain'
                . ' at least 1 Generator.');
        }

        foreach ($generators as $generator) {
            if (!$generator instanceof Generator) {
                throw new InvalidArgumentException('Generator must'
                    . ' implement the Generator interface.');
            }
        }

        $dimensions = current($generators)->dimensions();

        $k = count($generators);

        foreach ($generators as $generator) {
            if ($generator->dimensions() !== $dimensions) {
                throw new InvalidArgumentException('Agglomerate must contain'
                    . ' Generators that produce samples of the same'
                    . " dimensionality, $dimensions expected but "
                    . " {$generator->dimensions()} given.");
            }
        }

        if (is_array($weights)) {
            if (count($weights) !== $k) {
                throw new InvalidArgumentException('The number of weights'
                    . " and Generators must be equal, $k expected but "
                    . count($weights) . ' given.');
            }

            foreach ($weights as $weight) {
                if ($weight < 0) {
                    throw new InvalidArgumentException('Weights must be'
                        . " positive, $weight given.");
                }
            }

            $total = array_sum($weights);

            if ($total == 0) {
                throw new InvalidArgumentException('Total weight must'
                    . ' not be equal to 0.');
            }

            foreach ($weights as &$weight) {
                $weight /= $total;
            }
        } else {
            $weights = array_fill(0, $k, 1.0 / $k);
        }

        $this->generators = $generators;
        $this->weights = array_combine(array_keys($generators), $weights);
        $this->dimensions = $dimensions;
    }

    /**
     * Return the normalized weights of each generator in the agglomerate.
     *
     * @return (int|float)[]
     */
    public function weights() : array
    {
        return $this->weights;
    }

    /**
     * Return the dimensionality of the data this generates.
     *
     * @internal
     *
     * @return int
     */
    public function dimensions() : int
    {
        return $this->dimensions;
    }

    /**
     * Generate n data points.
     *
     * @param int $n
     * @return Labeled
     */
    public function generate(int $n) : Labeled
    {
        $samples = $labels = [];

        foreach ($this->generators as $label => $generator) {
            $p = (int) round($this->weights[$label] * $n);

            if ($p < 1) {
                continue;
            }

            $samples[] = $generator->generate($p)->samples();
            $labels[] = array_fill(0, $p, $label);
        }

        return Labeled::quick(
            $samples ? array_merge(...$samples) : [],
            $labels ? array_merge(...$labels) : []
        );
    }
}
