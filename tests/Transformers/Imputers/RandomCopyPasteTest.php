<?php

use Rubix\Engine\Transformers\Imputers\Imputer;
use Rubix\Engine\Transformers\Imputers\RandomCopyPaste;
use PHPUnit\Framework\TestCase;

class RandomCopyPasteTest extends TestCase
{
    protected $values;

    protected $imputer;

    public function setUp()
    {
        $this->values = [1, 2, 3, 4, 5];

        $this->imputer = new RandomCopyPaste();
    }

    public function test_build_random_copy_paste_imputer()
    {
        $this->assertInstanceOf(RandomCopyPaste::class, $this->imputer);
        $this->assertInstanceOf(Imputer::class, $this->imputer);
    }

    public function test_guess_value()
    {
        $this->imputer->fit($this->values);

        $value = $this->imputer->impute();

        $this->assertContains($value, $this->values);
    }
}
