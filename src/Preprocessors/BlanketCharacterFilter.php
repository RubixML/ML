<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;

class BlanketCharacterFilter implements Preprocessor
{
    const SPECIAL = [
        '.', ',', '?', '!', '#', '(', ')', '[', ']', '{', '}', ':', ';', '\'', '"',
        '|', '<', '>', '/', '\\', '+', '=',
    ];

    const NUMBERS = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ];

    /**
     * The characters to remove from the text.
     *
     * @var array
     */
    protected $remove = [
        //
    ];

    /**

     * @param  array|null  $remove
     * @return void
     */
    public function __construct(array $remove = self::SPECIAL)
    {
        $this->remove = $remove;
    }

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
                    $feature = str_replace($this->remove, '', $feature);
                }
            }
        }
    }
}
