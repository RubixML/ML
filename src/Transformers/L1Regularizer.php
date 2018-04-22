<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Dataset;

class L1Regularizer implements Transformer
{
    /**
     * The columns that should be regularized. i.e. the continuous data points.
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
     * Determine the columns that need to be regularized.
     *
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
     * Regularize the dataset by dividing each feature by the L1 norm of the sample
     * vector.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $norm = array_reduce($this->columns, function ($carry, $column) use ($sample) {
                return $carry += $sample[$column];
            }, 0);

            foreach ($this->columns as $column) {
                $sample[$column] /= ($norm ? $norm : self::EPSILON);
            }
        }
    }
}
