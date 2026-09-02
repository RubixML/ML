<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[RequiresPhpExtension('tensor')]
#[CoversClass(PrincipalComponentAnalysis::class)]
class PrincipalComponentAnalysisTest extends TestCase
{
    /**
     * @var Blob
     */
    protected $generator;

    /**
     * @var PrincipalComponentAnalysis
     */
    protected $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob([0.0, 3000.0, -6.0, 25], [1.0, 30.0, 0.001, 10.0]);

        $this->transformer = new PrincipalComponentAnalysis(2);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(PrincipalComponentAnalysis::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    #[Test]
    public function fitTransform() : void
    {
        $this->assertEquals(4, $this->generator->dimensions());

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(2, $sample);
    }

    #[Test]
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
