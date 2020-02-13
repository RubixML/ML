<?php

namespace Rubix\ML\Datasets\Generators;

use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

use function count;
use function gettype;

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
    protected $generators;

    /**
     * The normalized weights of each generator i.e. the probability that a
     * sample from a particular generator shows up in the dataset.
     *
     * @var float[]
     */
    protected $weights;

    /**
     * The dimensionality of the agglomerate.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * @param \Rubix\ML\Datasets\Generators\Generator[] $generators
     * @param (int|float)[]|null $weights
     * @throws \InvalidArgumentException
     */
    public function __construct(array $generators = [], ?array $weights = null)
    {
        if (empty($generators)) {
            throw new InvalidArgumentException('Agglomerate must consist'
                . ' of at least 1 generator.');
        }

        foreach ($generators as $generator) {
            if (!$generator instanceof Generator) {
                throw new InvalidArgumentException('Cannot add a non generator'
                    . ' to the agglomerate, ' . gettype($generator)
                    . ' given.');
            }
        }

        $dimensions = current($generators)->dimensions();

        $k = count($generators);

        foreach ($generators as $generator) {
            if ($generator->dimensions() !== $dimensions) {
                throw new InvalidArgumentException('Generators must have the'
                    . " same dimensionality, $dimensions needed but "
                    . " {$generator->dimensions()} given.");
            }
        }

        if (is_array($weights)) {
            if (count($weights) !== $k) {
                throw new InvalidArgumentException('The number of weights must'
                    . " equal the number of generators, $k needed but found "
                    . count($weights) . '.');
            }

            foreach ($weights as $weight) {
                if ($weight < 0) {
                    throw new InvalidArgumentException('Weights must all be'
                        . " positive, $weight found.");
                }
            }

            $total = array_sum($weights);

            if ($total == 0) {
                throw new InvalidArgumentException('Total weight for the'
                    . ' agglomerate cannot be 0.');
            }

            foreach ($weights as &$weight) {
                $weight /= $total;
            }
        } else {
            $weights = array_fill(0, $k, 1.0 / $k);
        }

        $this->generators = $generators;
        $this->weights = array_combine(array_keys($generators), $weights) ?: [];
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
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function generate(int $n) : Labeled
    {
        $dataset = Labeled::quick();

        foreach ($this->generators as $label => $generator) {
            $p = (int) round($this->weights[$label] * $n);

            if ($p < 1) {
                continue 1;
            }

            $samples = $generator->generate($p)->samples();

            $labels = array_fill(0, $p, $label);

            $dataset = $dataset->append(Labeled::quick($samples, $labels));
        }

        return $dataset;
    }
}
