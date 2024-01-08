<?php

namespace Rubix\ML\Backends\Swoole;

interface Serializer
{
    public function serialize(mixed $value) : string;

    public function unserialize(string $data) : mixed;
}
