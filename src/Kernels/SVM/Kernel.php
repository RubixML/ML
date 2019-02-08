<?php

namespace Rubix\ML\Kernels\SVM;

interface Kernel
{
    /**
     * Return the options for the libsvm runtime.
     *
     * @return array
     */
    public function options() : array;
}
