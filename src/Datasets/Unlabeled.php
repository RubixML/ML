<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Other\Structures\DataFrame;
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
     * Restore an unlabeled dataset from a serialized object file.
     *
     * @param  string  $path
     * @throws \RuntimeException
     * @return self
     */
    public static function restore(string $path) : self
    {
        if (!file_exists($path) or !is_readable($path)) {
            throw new RuntimeException('File ' . basename($path) . ' cannot be'
                . ' opened. Check path and permissions.');
        }

        $dataset = unserialize(file_get_contents($path) ?: '');

        if (!$dataset instanceof Unlabeled) {
            throw new RuntimeException('Dataset could not be reconstituted.');
        }

        return $dataset;
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
     * Return an array of autodetected feature column types.
     *
     * @return array
     */
    public function columnTypes() : array
    {
        return array_map(function ($feature) {
            return is_string($feature) ? self::CATEGORICAL : self::CONTINUOUS;
        }, $this->samples[0] ?? []);
    }

    /**
     * Get the column type for a given column index.
     *
     * @param  int  $index
     * @return int
     */
    public function type(int $index) : int
    {
        return is_string($this->samples[0][$index])
            ? self::CATEGORICAL : self::CONTINUOUS;
    }

    /**
     * Apply a transformation to the sample matrix.
     *
     * @param  \Rubix\ML\Transformers\Transformer  $transformer
     * @return void
     */
    public function apply(Transformer $transformer) : void
    {
        $transformer->transform($this->samples);
    }

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function head(int $n = 10) : self
    {
        return new self(array_slice($this->samples, 0, $n), false);
    }

    /**
     * Return a dataset containing only the last n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function tail(int $n = 10) : self
    {
        return new self(array_slice($this->samples, -$n), false);
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
     * Remove a size n chunk of the dataset starting at offset and return it in
     * a new dataset.
     *
     * @param  int  $offset
     * @param  int  $n
     * @return self
     */
    public function splice(int $offset, int $n) : self
    {
        return new self(array_splice($this->samples, $offset, $n), false);
    }

    /**
     * Randomize the dataset.
     *
     * @return self
     */
    public function randomize() : self
    {
        shuffle($this->samples);

        return $this;
    }

    /**
     * Sort the dataset by a column in the sample matrix.
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
            . ' between 0 and 1.');
        }

        $n = (int) ($ratio * $this->numRows());

        $left = new self(array_slice($this->samples, 0, $n), false);
        $right = new self(array_slice($this->samples, $n), false);

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
            throw new InvalidArgumentException('Cannot fold the dataset less'
                . ' than 1 time.');
        }

        $samples = $this->samples;

        $n = (int) floor(count($samples) / $k);

        $folds = [];

        for ($i = 0; $i < $k; $i++) {
            $folds[] = new self(array_splice($samples, 0, $n), false);
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
            $batches[] = new self($batch, false);
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
     * @return array
     */
    public function partition(int $index, $value) : array
    {
        if (!is_string($value) and !is_numeric($value)) {
            throw new InvalidArgumentException('Value must be a string or'
                . ' numeric type.');
        }

        $left = $right = [];

        if ($this->type($index) === self::CATEGORICAL) {
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

        return [new self($left, false), new self($right, false)];
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
                . ' less than 1 sample.');
        }

        $max = $this->numRows() - 1;

        $subset = [];

        for ($i = 0; $i < $n; $i++) {
            $subset[] = $this->samples[rand(0, $max)];
        }

        return new self($subset, false);
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
                . ' equals to the number of samples in the dataset.');
        }

        $total = array_sum($weights);
        $max = (int) round($total * self::PHI);

        $subset = [];

        for ($i = 0; $i < $n; $i++) {
            $delta = rand(0, $max) / self::PHI;

            foreach ($weights as $row => $weight) {
                $delta -= $weight;

                if ($delta < 0.) {
                    $subset[] = $this->samples[$row];
                    break 1;
                }
            }
        }

        return new self($subset, false);
    }

    /**
     * Save the dataset to a serialized object file.
     *
     * @param  string|null  $path
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function save(?string $path = null) : void
    {
        if (is_null($path)) {
            $path = (string) time() . '.dataset';
        }

        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not'
                . ' writable. Check path and permissions.');
        }

        $success = file_put_contents($path, serialize($this), LOCK_EX);

        if (!$success) {
            throw new RuntimeException('Failed to serialize object to storage.');
        }
    }

    /**
     * Append the given dataset to the end of this dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function prepend(Dataset $dataset) : Dataset
    {
        $this->samples = array_merge($dataset->samples(), $this->samples);

        return $this;
    }

    /**
     * Append the given dataset to the end of this dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function append(Dataset $dataset) : Dataset
    {
        $this->samples = array_merge($this->samples, $dataset->samples());

        return $this;
    }
}
