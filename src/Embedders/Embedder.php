<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\Transformers\Transformer;

/**
 * Embedder
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Embedder extends Transformer
{
    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array;
}
