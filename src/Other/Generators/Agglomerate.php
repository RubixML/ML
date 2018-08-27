<?php

namespace Rubix\ML\Other\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * Agglomerate
 *
 * An Agglomerate is a collection of other generators each assigned a label.
 * Agglomerates are useful for classification, clustering, and anomaly detection
 * problems where the label is a discrete value.
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
     * @var array
     */
    protected $generators;

    /**
     * The normalized weights of each generator i.e. the probability that a
     * sample from a particular generator shows up in the dataset.
     *
     * @var array
     */
    protected $weights;

    /**
     * The number of dimensions of the agglomerate.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * @param  array  $generators
     * @param  array|null  $weights
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $generators = [], ?array $weights = null)
    {
        $n = count($generators);

        if ($n === 0) {
            throw new InvalidArgumentException('Must provide at least one'
                . ' generator to agglomerate.');
        }

        $dimensions = current($generators)->dimensions();

        foreach ($generators as $label => $generator) {
            if (!$generator instanceof Generator) {
                throw new InvalidArgumentException('Non generator object found.');
            }

            if ($generator->dimensions() !== $dimensions) {
                throw new InvalidArgumentException('Generators must each be'
                    . ' the same dimension.');
            }
        }

        if (is_array($weights)) {
            if (count($weights) !== $n) {
                throw new InvalidArgumentException('The number of weights must'
                    . ' the number of generators.');
            }

            $total = array_sum($weights);

            if ($total === 0.) {
                throw new InvalidArgumentException('Total weight for the'
                    . ' agglomerate cannot be 0.');
            }

            $weights = array_map(function ($value) use ($total) {
                return $value / $total;
            }, $weights);
        } else {
            $weights = array_fill(0, $n, 1. / $n);
        }

        $this->generators = $generators;
        $this->weights = array_combine(array_keys($generators), $weights);
        $this->dimensions = $dimensions;
    }

    /**
     * Return the normalized weights of each generator in the agglomerate.
     *
     * @return array
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
     * @param  int  $n
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n = 100) : Dataset
    {
        $dataset = new Labeled();

        foreach ($this->generators as $label => $generator) {
            $k = (int) round($this->weights[$label] * $n);

            $samples = $generator->generate($k)->samples();

            $labels = array_fill(0, $k, $label);

            $dataset->append(new Labeled($samples, $labels, false));
        }

        return $dataset;
    }
}
