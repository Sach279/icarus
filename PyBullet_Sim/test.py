def check_robot_collision(self, base_pos, all_robots):
        for other_bot in all_robots:
            if other_bot.robot_id == self.robot_id: continue
            try:
                other_pos, _ = p.getBasePositionAndOrientation(other_bot.robot_id)
                if not are_valid_coordinates(other_pos): continue
                dist_sq = (base_pos[0] - other_pos[0])**2 + (base_pos[1] - other_pos[1])**2
                if dist_sq < ROBOT_COLLISION_THRESHOLD**2:
                    if self.robot_instance_id > other_bot.robot_instance_id:
                        return True
            except p.error: continue
        return False