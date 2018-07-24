<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Datasets\Structures\DataFrame;
use InvalidArgumentException;
use RuntimeException;

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
     * Return a dataset containing only the first n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function head(int $n = 10) : self
    {
        return new self(array_slice($this->samples, 0, $n));
    }

    /**
     * Return a dataset containing only the last n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function tail(int $n = 10) : self
    {
        return new self(array_slice($this->samples, -$n));
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
        return new self(array_splice($this->samples, $offset, $n));
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
        if ($ratio <= 0 or $ratio >= 1) {
            throw new InvalidArgumentException('Split ratio must be strictly'
            . ' between 0 and 1.');
        }

        $n = (int) ($ratio * $this->numRows());

        return [
            new self(array_slice($this->samples, 0, $n)),
            new self(array_slice($this->samples, $n)),
        ];
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

        $n = (int) (count($samples) / $k);

        $folds = [];

        for ($i = 0; $i < $k; $i++) {
            $folds[] = new self(array_splice($samples, 0, $n));
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

        while (!empty($samples)) {
            $batches[] = new self(array_splice($samples, 0, $n));
        }

        return $batches;
    }

    /**
     * Generate a random subset.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubset(int $n = 1) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . ' less than 1 sample.');
        }

        if ($n > $this->numRows()) {
            throw new InvalidArgumentException('Cannot generate a larger subset'
                . ' than the sample size.');
        }

        return new self(array_intersect_key($this->samples,
            (array) array_rand($this->samples, $n)));
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubsetWithReplacement(int $n = 1) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . ' less than 1 sample.');
        }

        $subset = [];

        for ($i = 0; $i < $n; $i++) {
            $subset[] = $this->samples[array_rand($this->samples)];
        }

        return new self($subset);
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
