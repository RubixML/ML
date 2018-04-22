<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Dataset;

class TextNormalizer implements Transformer
{
    /**
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void
    {
        //
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
            foreach ($sample as &$feature) {
                if (is_string($feature)) {
                    $feature = strtolower(preg_replace('/\s+/', ' ', trim($feature)));
                }
            }
        }
    }
}
