<?php

namespace Rubix\ML\Kernels\SVM;

use Stringable;

interface Kernel extends Stringable
{
    /**
     * Return the options for the libsvm runtime.
     *
     * @return mixed[]
     */
    public function options() : array;
}
