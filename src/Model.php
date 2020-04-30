<?php

namespace Rubix\ML;

interface Model
{
    /**
     * Return the data types that the model is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array;

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array;
}
