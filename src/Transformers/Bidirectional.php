<?php

namespace Rubix\ML\Transformers;

/**
 * Bidirectional
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Alex Torchenko
 */
interface Bidirectional extends Transformer
{
    /**
     * Reverse transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     */
    public function reverseTransform(array &$samples) : void;
}
