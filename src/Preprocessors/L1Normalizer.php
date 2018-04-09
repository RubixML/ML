<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;

class L1Normalizer implements Preprocessor
{
    /**
     * The columns that should be normalized. i.e. the continuous data points.
     *
     * @var array
     */
    protected $columns = [
        //
    ];

    /**
     * @return array
     */
    public function columns() : array
    {
        return $this->columns;
    }

    /**
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void
    {
        $this->columns = [];

        foreach ($data->columnTypes() as $column => $type) {
            if ($type === self::CONTINUOUS) {
                $this->columns[] = $column;
            }
        }
    }

    /**
     * Normalize the dataset.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $norm = array_reduce($this->columns, function ($carry, $column) use ($sample) {
                return $carry += $sample[$column];
            }, 0) + self::EPSILON;

            foreach ($this->columns as $column) {
                $sample[$column] /= $norm;
            }
        }
    }
}
