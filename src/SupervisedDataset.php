<?php

namespace Rubix\Engine;

use Rubix\Engine\Preprocessors\Preprocessor;
use InvalidArgumentException;
use Countable;

class SupervisedDataset implements Countable
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    /**
     * The feature vectors or columns of a data table.
     *
     * @var array
     */
    protected $samples;

    /**
     * The labeled outcomes used for supervised training.
     *
     * @var array
     */
    protected $outcomes;

    /**
     * Build a supervised dataset used for training and testing models. The assumption
     * is the that dataset contain 0 < n < ∞ feature columns where the last column is
     * always the labeled outcome.
     *
     * @param  iterable  $data
     * @return self
     */
    public static function build(iterable $data) : self
    {
        $samples = $outcomes = [];

        foreach ($data as $row) {
            $outcomes[] = array_pop($row);
            $samples[] = array_values($row);
        }

        return new static($samples, $outcomes);
    }

    /**
     * @param  array  $samples
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $samples, array $outcomes)
    {
        if (count($samples) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of samples must equal the number of outcomes.');
        }

        foreach ($samples as &$sample) {
            if (count($sample) !== count($samples[0])) {
                throw new InvalidArgumentException('The number of feature columns must be equal for all samples.');
            }

            foreach ($sample as &$feature) {
                if (!is_string($feature) && !is_numeric($feature)) {
                    throw new InvalidArgumentException('Feature values must be a string or numeric type, ' . gettype($feature) . ' found.');
                }

                if (is_string($feature) && is_numeric($feature)) {
                    if (is_float($feature + 0)) {
                        $feature = (float) $feature;
                    } else {
                        $feature = (int) $feature;
                    }
                }
            }
        }

        $this->samples = $samples;
        $this->outcomes = $outcomes;
    }

    /**
     * @return array
     */
    public function samples() : array
    {
        return $this->samples;
    }

    /**
     * @return int
     */
    public function rows() : int
    {
        return count($this->samples);
    }

    /**
     * The number of feature columns in this dataset.
     *
     * @return int
     */
    public function columns() : int
    {
        return count($this->samples[0] ?? []);
    }

    /**
     * Return the types for the feature columns.
     *
     * @return array
     */
    public function columnTypes() : array
    {
        return array_map(function ($feature) {
            return is_string($feature) ? self::CATEGORICAL : self::CONTINUOUS;
        }, $this->samples[0]);
    }

    /**
     * @return array
     */
    public function outcomes() : array
    {
        return $this->outcomes;
    }

    /**
     * The set of all possible labeled outcomes.
     *
     * @return array
     */
    public function labels() : array
    {
        return array_unique($this->outcomes);
    }

    /**
     * Have a preprocessor transform the dataset.
     *
     * @param  \Rubix\Engine\Preprocessors\Preprocessor  $preprocessor
     * @return self
     */
    public function transform(Preprocessor $preprocessor) : self
    {
        $preprocessor->transform($this->samples);

        return $this;
    }

    /**
     * Randomize the dataset.
     *
     * @return self
     */
    public function randomize() : self
    {
        $order = range(0, count($this->outcomes) - 1);

        shuffle($order);

        array_multisort($order, $this->samples, $this->outcomes);

        return $this;
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @return array
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0.0 || $ratio >= 0.9) {
            throw new InvalidArgumentException('Split ratio must be a float value between 0.0 and 0.9.');
        }

        $strata = $this->stratify();

        $training = $testing = [0 => [], 1 => []];

        foreach ($strata[0] as $i => $stratum) {
            $testing[0] = array_merge($testing[0], array_splice($stratum, 0, round($ratio * count($stratum))));
            $testing[1] = array_merge($testing[1], array_splice($strata[1][$i], 0, round($ratio * count($strata[1][$i]))));

            $training[0] = array_merge($training[0], $stratum);
            $training[1] = array_merge($training[1], $strata[1][$i]);
        }

        return [
            new static(...$training),
            new static(...$testing),
        ];
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  float  $ratio
     * @return self
     */
    public function generateRandomSubset(float $ratio = 0.1) : self
    {
        $n = ceil($ratio * $this->rows());
        $max = $this->rows() - 1;
        $subset = [];

        foreach (range(1, $n) as $i) {
            $index = random_int(0, $max);

            $subset[0][] = $this->samples[$index];
            $subset[1][] = $this->outcomes[$index];
        }

        return new static(...$subset);
    }

    /**
     * Take n samples and outcomes from this dataset and return them in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function take(int $n = 1) : self
    {
        return new static(array_splice($this->samples, 0, $n), array_splice($this->outcomes, 0, $n));
    }

    /**
     * Remove a feature column from the dataset given by the column's offset.
     *
     * @param  int  $offset
     * @return self
     */
    public function removeColumn(int $offset) : self
    {
        foreach ($this->samples as &$sample) {
            unset($sample[$offset]);

            $sample = array_values($sample);
        }
    }

    /**
     * Group samples by outcome and return an array of strata.
     *
     * @return array
     */
    public function stratify() : array
    {
        $strata = [];

        foreach ($this->outcomes as $i => $outcome) {
            $strata[0][$outcome][] = $this->samples[$i];
            $strata[1][$outcome][] = $outcome;
        }

        return $strata;
    }

    /**
     * @return array
     */
    public function toArray() : array
    {
        return [
            $this->samples,
            $this->outcomes,
        ];
    }

    /**
     * @return int
     */
    public function count() : int
    {
        return $this->rows();
    }
}
