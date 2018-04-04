<?php

namespace Rubix\Engine;

use InvalidArgumentException;
use Countable;

class SupervisedDataset implements Countable
{
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
     * @param  array  $data
     * @return void
     */
    public function __construct(array $data)
    {
        foreach ($data as &$sample) {
            if (!is_array($sample)) {
                $sample = [$sample];
            }

            foreach ($sample as &$feature) {
                if (is_string($feature) && is_numeric($feature)) {
                    if (is_float($feature + 0)) {
                        $feature = (float) $feature;
                    } else {
                        $feature = (int) $feature;
                    }
                }
            }

            $this->outcomes[] = array_pop($sample);

            if (count($sample) !== count($data[0])) {
                throw new InvalidArgumentException('The number of feature columns must be equal for all samples.');
            }

            $this->samples[] = array_values($sample);
        }
    }

    /**
     * @return array
     */
    public function samples() : array
    {
        return $this->samples;
    }

    /**
     * @return array
     */
    public function outcomes() : array
    {
        return $this->outcomes;
    }

    /**
     * All possible labels of the outcomes.
     *
     * @return array
     */
    public function labels() : array
    {
        return array_unique($this->outcomes);
    }

    /**
     * @return int
     */
    public function count() : int
    {
        return count($this->samples);
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
     * Split the dataset into two stratified subsets.
     *
     * @param  float  $ratio
     * @return array
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0.0 || $ratio >= 0.9) {
            throw new InvalidArgumentException('Split ratio must be a float value between 0.0 and 0.9.');
        }

        $strata = $this->stratify($this->samples, $this->outcomes);

        $training = [];
        $testing = [];

        foreach ($strata as $stratum) {
            $testing = array_merge($testing, array_splice($stratum, 0, round($ratio * count($stratum))));
            $training = array_merge($training, $stratum);
        }

        return [
            new static($training),
            new static($testing),
        ];
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
     * @param  array  $samples
     * @param  array  $outcomes
     * @return array
     */
    protected function stratify(array $samples, array $outcomes) : array
    {
        $classes = array_unique($outcomes);

        $strata = array_combine($classes, array_fill(0, count($classes), []));

        foreach ($outcomes as $i => $outcome) {
            $strata[$outcome][] = array_merge($samples[$i], [$outcome]);
        }

        return $strata;
    }
}
