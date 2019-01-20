<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Transformers\Transformer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Unlabeled
 *
 * Unlabeled datasets can be used to train *unsupervised* Estimators and for
 * feeding data into an Estimator to make predictions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Unlabeled extends DataFrame implements Dataset
{
    /**
     * Build a new unlabeled dataset with validation.
     * 
     * @param  array  $samples
     * @return self
     */
    public static function build(array $samples = []) : self
    {
        return new self($samples, true);
    }

    /**
     * Build a new unlabeled dataset foregoing validation.
     * 
     * @param  array[]  $samples
     * @return self
     */
    public static function quick(array $samples = []) : self
    {
        return new self($samples, false);
    }

    /**
     * Build a dataset from an iterator.
     * 
     * @param  iterable  $samples
     * @return self
     */
    public static function fromIterator(iterable $samples) : self
    {
        $samples = is_array($samples)
            ? $samples
            : iterator_to_array($samples, false);

        return self::build($samples);
    }

    /**
     * Stack a number of datasets on top of each other to form a single
     * dataset.
     * 
     * @param  array  $datasets
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function stack(array $datasets) : self
    {
        $samples = $labels = [];

        foreach ($datasets as $dataset) {
            if (!$dataset instanceof Dataset) {
                throw new InvalidArgumentException('Dataset must be'
                    . ' an instance of Dataset, ' . get_class($dataset)
                    . ' given.');
            }

            $samples = array_merge($samples, $dataset->samples());
        }

        return self::quick($samples);
    }

    /**
     * @param  array  $samples
     * @param  bool  $validate
     * @return void
     */
    public function __construct(array $samples = [], bool $validate = true)
    {
        parent::__construct($samples, $validate);
    }

    /**
     * Apply a transformation to the dataset and return for chaining.
     *
     * @param  \Rubix\ML\Transformers\Transformer  $transformer
     * @return self
     */
    public function apply(Transformer $transformer) : self
    {
        $transformer->transform($this->samples);

        return $this;
    }

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function head(int $n = 10) : self
    {
        return self::quick(array_slice($this->samples, 0, $n));
    }

    /**
     * Return a dataset containing only the last n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function tail(int $n = 10) : self
    {
        return self::quick(array_slice($this->samples, -$n));
    }

    /**
     * Take n samples from this dataset and return them in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function take(int $n = 1) : self
    {
        return $this->splice(0, $n);
    }

    /**
     * Leave n samples on this dataset and return the rest in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function leave(int $n = 1) : self
    {
        return $this->splice($n, $this->numRows());
    }

    /**
     * Prepend this dataset with another dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function prepend(Dataset $dataset) : Dataset
    {
        return self::quick(array_merge($dataset->samples(), $this->samples));
    }

    /**
     * Append this dataset with another dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function append(Dataset $dataset) : Dataset
    {
        return self::quick(array_merge($this->samples, $dataset->samples()));
    }

    /**
     * Remove a size n chunk of the dataset starting at offset and return it in
     * a new dataset.
     *
     * @param  int  $offset
     * @param  int  $n
     * @return self
     */
    public function splice(int $offset, int $n) : self
    {
        return self::quick(array_splice($this->samples, $offset, $n));
    }

    /**
     * Randomize the dataset in place and return self for chaining.
     *
     * @return self
     */
    public function randomize() : self
    {
        shuffle($this->samples);

        return $this;
    }

    /**
     * Run a filter over the dataset using the values of a given column.
     *
     * @param  int  $index
     * @param  callable  $fn
     * @return self
     */
    public function filterByColumn(int $index, callable $fn) : self
    {
        $samples = [];

        foreach ($this->samples as $sample) {
            if ($fn($sample[$index])) {
                $samples[] = $sample;
            }
        }

        return self::quick($samples);
    }

    /**
     * Sort the dataset in place by a column in the sample matrix.
     *
     * @param  int  $index
     * @param  bool  $descending
     * @return self
     */
    public function sortByColumn(int $index, bool $descending = false)
    {
        $order = $this->column($index);

        array_multisort($order, $this->samples, $descending ? SORT_DESC : SORT_ASC);

        return $this;
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return array
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0. or $ratio >= 1.) {
            throw new InvalidArgumentException('Split ratio must be strictly'
                . " between 0 and 1, $ratio given.");
        }

        $p = (int) ($ratio * $this->numRows());

        $left = self::quick(array_slice($this->samples, 0, $p));
        $right = self::quick(array_slice($this->samples, $p));

        return [$left, $right];
    }

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return array
     */
    public function fold(int $k = 3) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot create less than 2'
                . " folds, $k given.");
        }

        $samples = $this->samples;

        $n = (int) floor(count($samples) / $k);

        $folds = [];

        for ($i = 0; $i < $k; $i++) {
            $folds[] = self::quick(array_splice($samples, 0, $n));
        }

        return $folds;
    }

    /**
     * Generate a collection of batches of size n from the dataset. If there are
     * not enough samples to fill an entire batch, then the dataset will contain
     * as many samples as possible.
     *
     * @param  int  $n
     * @return array
     */
    public function batch(int $n = 50) : array
    {
        $batches = [];

        $samples = $this->samples;

        foreach (array_chunk($this->samples, $n) as $batch) {
            $batches[] = self::quick($batch);
        }

        return $batches;
    }

    /**
     * Partition the dataset into left and right subsets by a specified feature
     * column. The dataset is split such that, for categorical values, the left
     * subset contains all samples that match the value and the right side
     * contains samples that do not match. For continuous values, the left side
     * contains all the  samples that are less than the target value, and the
     * right side contains the samples that are greater than or equal to the
     * value.
     *
     * @param  int  $index
     * @param  mixed  $value
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function partition(int $index, $value) : array
    {
        if (!is_string($value) and !is_numeric($value)) {
            throw new InvalidArgumentException('Value must be a string or'
                . ' numeric type, ' . gettype($value) . ' given.');
        }

        $left = $right = [];

        if ($this->columnType($index) === DataFrame::CATEGORICAL) {
            foreach ($this->samples as $i => $sample) {
                if ($sample[$index] === $value) {
                    $left[] = $sample;
                } else {
                    $right[] = $sample;
                }
            }
        } else {
            foreach ($this->samples as $i => $sample) {
                if ($sample[$index] < $value) {
                    $left[] = $sample;
                } else {
                    $right[] = $sample;
                }
            }
        }

        return [self::quick($left), self::quick($right)];
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubsetWithReplacement(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . " less than 1 sample, $n given.");
        }

        $max = $this->numRows() - 1;

        $subset = [];

        for ($i = 0; $i < $n; $i++) {
            $subset[] = $this->samples[rand(0, $max)];
        }

        return self::quick($subset);
    }

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param  int  $n
     * @param  array  $weights
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomWeightedSubsetWithReplacement(int $n, array $weights) : self
    {
        if (count($weights) !== count($this->samples)) {
            throw new InvalidArgumentException('The number of weights must be'
                . ' equal to the number of samples in the dataset, '
                . count($this->samples) . ' needed, ' . count($weights)
                . ' given.');
        }

        $total = array_sum($weights);
        $max = (int) round($total * self::PHI);

        $subset = [];

        for ($i = 0; $i < $n; $i++) {
            $delta = rand(0, $max) / self::PHI;

            foreach ($weights as $row => $weight) {
                $delta -= $weight;

                if ($delta <= 0.) {
                    $subset[] = $this->samples[$row];
                    break 1;
                }
            }
        }

        return self::quick($subset);
    }

    /**
     * Specify data which should be serialized to JSON.
     * 
     * @return mixed
     */
    public function jsonSerialize()
    {
        return [
            'samples' => $this->samples,
        ];
    }
}
