<?php

namespace Rubix\ML\Tests\Persisters;

use Rubix\ML\Persisters\RedisDB;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Classifiers\DummyClassifier;
use PHPUnit\Framework\TestCase;

/**
 * @group Persisters
 * @requires extension redis
 * @covers \Rubix\ML\Persisters\RedisDB
 */
class RedisDBTest extends TestCase
{
    /**
     * @var \Rubix\ML\Persistable
     */
    protected $persistable;

    /**
     * @var \Rubix\ML\Persisters\RedisDB
     */
    protected $persister;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->persistable = new DummyClassifier();

        $this->persister = $this->createMock(RedisDB::class);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RedisDB::class, $this->persister);
        $this->assertInstanceOf(Persister::class, $this->persister);
    }
}
