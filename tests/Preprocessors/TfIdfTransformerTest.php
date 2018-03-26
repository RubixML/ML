<?php

use Rubix\Engine\Preprocessors\TfIdfTransformer;
use PHPUnit\Framework\TestCase;

class TfIdfTransformerTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $this->preprocessor = new TfIdfTransformer();
    }

    public function test_build_tf_idf_transformer()
    {
        $this->assertInstanceOf(TfIdfTransformer::class, $this->preprocessor);
    }
}
