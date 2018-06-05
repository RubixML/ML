<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;

class TextNormalizer implements Transformer
{
    /**
     * Should the text be converted to all lowercase?
     *
     * @var bool
     */
    protected $lowercase;

    /**
     * @param  bool  $lowercase
     * @param  bool  $whitespace
     * @return void
     */
    public function __construct(bool $lowercase = true)
    {
        $this->lowercase = $lowercase;
    }

    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
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
                    if ($this->lowercase) {
                        $feature = strtolower($feature);
                    }

                    $feature = preg_replace('/\s+/', ' ', trim($feature));
                }
            }
        }
    }
}
