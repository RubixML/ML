<?php

namespace Rubix\ML\Transformers;

/**
 * Reversible
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Alex Torchenko
 */
interface Reversible extends Transformer
{
    /**
     * Perform the reverse transformation to the samples.
     *
     * @param list<list<mixed>> $samples
     */
    public function reverseTransform(array &$samples) : void;
}
