<?php

namespace Rubix\Engine\Transformers\Imputers;

class RandomCopyPaste implements Continuous
{
    /**
     * The memoized values of the fitted feature column.
     *
     * @var array
     */
    protected $values = [
        //
    ];

    /**
     * Copy the values.
     *
     * @param  array  $values
     * @return void
     */
    public function fit(array $values) : void
    {
        $this->values = $values;
    }

    /**
     * Chose a random value from the fitted dataset and return it.
     *
     * @return mixed
     */
    public function impute()
    {
        return $this->values[array_rand($this->values)];
    }
}
