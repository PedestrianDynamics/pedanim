from manim import *
import numpy as np

def clip_vector(vec, max_length):
    norm = np.linalg.norm(vec)
    return vec if norm <= max_length else vec * (max_length / norm)

class SocialForceModel(Scene):
    # Parameters
    MASS = 5
    RADIUS = 0.3
    DESIRED_SPEED = 1.5
    RELAXATION_TIME = 0.5
    A = 2.0
    B = 0.1
    DT = 0.1
    TOTAL_TIME = 10
   

    def construct(self):
        self.camera.background_color = WHITE
        self.setup_scene()
        self.simulate()

    def setup_scene(self):
        arrow_config = {
            "buff": 0,
            "tip_length": 10.3,
            "stroke_width": 4,
            "tip_shape": ArrowTriangleTip,  # or ArrowSquareTip, ArrowCircleTip
        }
        # Environment setup
        corridor_width = 2
        corridor_length = 14
        self.goal_pos = np.array([corridor_length / 2 - 1, 0, 0])
        self.agent_pos = np.array([-corridor_length / 2 + 1, 0.0, 0])
        self.agent_vel = np.array([0.5, 0, 0])

        # Corridor walls
        top = Line([-corridor_length/2, corridor_width/2, 0], [corridor_length/2, corridor_width/2, 0], color=GRAY)
        bottom = Line([-corridor_length/2, -corridor_width/2, 0], [corridor_length/2, -corridor_width/2, 0], color=GRAY)

        # Goal and labels
        self.goal_dot = Circle(radius=self.RADIUS, color=GREEN, fill_opacity=0.6).move_to(self.goal_pos)

        goal_label = Text("Goal", font_size=24, color=BLACK).next_to(self.goal_dot, DOWN)

        # Obstacle
        self.obstacle = Square(side_length=1, color=RED, fill_color=RED, fill_opacity=0.6).move_to([0, 0, 0])
        self.obstacle_label = Text("Obstacle", font_size=24, color=BLACK).next_to(self.obstacle, DOWN).shift(0.1)

        # Agent and label
        self.agent_visual = Circle(radius=self.RADIUS, color=BLUE, fill_opacity=0.6).move_to(self.agent_pos)
        #Dot(point=self.agent_pos, color=BLUE, radius=self.RADIUS)
        self.agent_label = Text("Agent", font_size=24, color=BLACK).next_to(self.agent_visual, DOWN)

        # Force arrows
        self.offset_up = np.array([0, 0.15, 0])
        self.offset_down = np.array([0, -0.15, 0])
        self.offset_mid = np.array([0, 0, 0])
        zero = np.array([0, 0, 0])

        self.driving_force_arrow = Arrow(self.agent_pos, self.agent_pos, color=GREEN, **arrow_config)
        self.repulsive_force_arrow = Arrow(self.agent_pos, self.agent_pos, color=RED, **arrow_config)
        self.total_force_arrow = Arrow(self.agent_pos, self.agent_pos, color=BLACK, **arrow_config)

        # Title
        title = Text("Social Force Model: Superposition", font_size=30, color=BLACK).to_edge(UP+LEFT)
        eq_drv = MathTex(
            r"\vec{F}_i^{\mathrm{drv}}=\frac{v_i^0\vec{e}_i^0 - \vec{v}_i}{\tau}",
            font_size=26,
            color=BLACK
        )
        eq_rep = MathTex(
            r"\vec{F}_{ij}^{\mathrm{rep}} = A_i \exp\Big( \frac{r_{ij} - d_{ij}}{B_i}\Big)"
            ,font_size=26, color=BLACK
        )
        eqs = VGroup(eq_drv, eq_rep).arrange(RIGHT, buff=1).to_edge(UP+LEFT, buff=1)

        # Force value display setup
        self.driving_value = DecimalNumber(0.0, num_decimal_places=2, color=GREEN)
        self.repulsive_value = DecimalNumber(0.0, num_decimal_places=2, color=RED)
        self.total_value = DecimalNumber(0.0, num_decimal_places=2, color=BLACK)
        self.driving_vector_text = MathTex(r"\vec{F}_{\text{drv}} = (0.00,\ 0.00)", font_size=20, color=GREEN)
        self.repulsive_vector_text = MathTex(r"\vec{F}_{\text{rep}} = (0.00,\ 0.00)", font_size=20, color=RED)
        self.total_vector_text = MathTex(r"\vec{F}_{\text{tot}} = (0.00,\ 0.00)", font_size=20, color=BLACK)

        # Layout
        force_labels = VGroup(
            self.driving_vector_text,
            self.repulsive_vector_text,
            self.total_vector_text,
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UR)

        vector_lines = VGroup(
            self.driving_vector_text,
            self.repulsive_vector_text,
            self.total_vector_text,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        vector_box = SurroundingRectangle(
            vector_lines,
            color=BLUE,
            corner_radius=0.15,
            stroke_width=2,
            fill_color=WHITE,
            fill_opacity=0.8,
            buff=0.3
        )

        vector_block = VGroup(vector_box, vector_lines).to_corner(UR)
        
        row1 = VGroup(
            MathTex(f"M = {self.MASS}", font_size=20, color=BLACK),
            MathTex(f"R = {self.RADIUS}", font_size=20, color=BLACK),
        ).arrange(RIGHT, buff=0.3)
        
        row2 = VGroup(
            MathTex(f"v^0 = {self.DESIRED_SPEED}", font_size=20, color=BLACK),
            MathTex(rf"\tau = {self.RELAXATION_TIME}", font_size=20, color=BLACK)
        ).arrange(RIGHT, buff=0.3)
        
        row3 = VGroup(
            MathTex(f"A = {self.A}", font_size=20, color=BLACK),
            MathTex(f"B = {self.B}", font_size=20, color=BLACK)
        ).arrange(RIGHT, buff=0.3)
        
        # Stack rows
        param_lines = VGroup(row1, row2, row3).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        
        param_box = SurroundingRectangle(
            param_lines,
            color=BLUE,
            corner_radius=0.15,
            stroke_width=2,
            fill_color=WHITE,
            fill_opacity=0.8,
            buff=0.3
        )
        

        self.zero_force_notice = Text("Net Force ≈ 0", font_size=28, color=BLACK).next_to(bottom, DOWN)
        self.zero_force_notice.set_opacity(0)  # initially hidden
        self.add(self.zero_force_notice)
        
        param_block = VGroup(param_box, param_lines)
        param_block.next_to(eqs, RIGHT, buff=0.8)
        
        self.add(
            title,
            eqs,
        )
        

        # Add to scene
        self.play(GrowFromCenter(self.agent_visual))
        self.play(GrowFromCenter(self.goal_dot))

        elements = VGroup(
            top, bottom,
            goal_label,
            self.obstacle,
            self.obstacle_label,
            self.agent_label,
            
        )
        self.play(FadeIn(elements, shift=UP, scale=0.95), run_time=1.5)

        self.add(vector_block)
        self.add(param_block)
      

    def get_desired_direction(self):
        diff = self.goal_pos - self.agent_pos
        norm = np.linalg.norm(diff)
        return diff / norm if norm > 1e-10 else np.zeros(3)

    def driving_force(self):
        desired_velocity = self.DESIRED_SPEED * self.get_desired_direction()
        return (desired_velocity - self.agent_vel) / self.RELAXATION_TIME

    def repulsive_force(self):
        obs_center = self.obstacle.get_center()
        diff = self.agent_pos - obs_center
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            return np.zeros(3)
        direction = diff / dist
        r_ij = 4*self.RADIUS# + self.obstacle.width / 2
        force_mag = self.A * np.exp((r_ij - dist) / self.B)
        return force_mag * direction



    def update_agent(self):
        f_drv = self.driving_force()
        f_rep = self.repulsive_force()
        f_tot = f_drv + f_rep
        acc = f_tot / self.MASS
        self.agent_vel += acc * self.DT
        self.agent_pos += self.agent_vel * self.DT
        return f_drv, f_rep, f_tot
    def update_force_vectors(self, f_drv, f_rep, f_tot, scale=1):
        max_arrow_length = 4.0
        scaled_drv = clip_vector(scale * f_drv, max_arrow_length)
        scaled_rep = clip_vector(scale * f_rep, max_arrow_length)
        scaled_tot = clip_vector(scale * f_tot, max_arrow_length)

        pos = self.agent_pos
        # Remove old arrows
        self.remove(self.driving_force_arrow, self.repulsive_force_arrow, self.total_force_arrow)

        # Create new lines with arrowheads
        self.driving_force_arrow = Line(
            start=pos + self.offset_up,
            end=pos + self.offset_up + scale * scaled_drv,
            color=GREEN,
        ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)

        self.repulsive_force_arrow = Line(
            start=pos + self.offset_down,
            end=pos + self.offset_down + scale * scaled_rep,
            color=RED,
        ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)

        self.total_force_arrow = Line(
            start=pos + self.offset_mid,
            end=pos + self.offset_mid + scale * scaled_tot,
            color=BLACK,
        ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)

        # Re-add to scene
        self.add(
            self.driving_force_arrow,
            self.repulsive_force_arrow,
            self.total_force_arrow,
    )

    def simulate(self):
        steps = int(self.TOTAL_TIME / self.DT)
        count = 0
        for step in range(steps):
            f_drv, f_rep, f_tot = self.update_agent()
            self.agent_visual.move_to(self.agent_pos)
            new_agent_label = Text("Agent", font_size=24).next_to(self.agent_visual, DOWN)
            self.agent_label.move_to(new_agent_label)
            self.update_force_vectors(f_drv, f_rep, f_tot)
            self.driving_value.set_value(np.linalg.norm(f_drv))
            self.repulsive_value.set_value(np.linalg.norm(f_rep))
            self.total_value.set_value(np.linalg.norm(f_tot))
            def fmt(v):
                return r"({:+.2f},\ {:+.2f})".format(v[0], v[1])
                
            
            self.driving_vector_text.become(
                MathTex(rf"\vec{{F}}_{{\text{{drv}}}} = {fmt(f_drv)}", font_size=20, color=GREEN).move_to(self.driving_vector_text)
            )
            self.repulsive_vector_text.become(
                MathTex(rf"\vec{{F}}_{{\text{{rep}}}} = {fmt(f_rep)}", font_size=20, color=RED).move_to(self.repulsive_vector_text)
            )
            self.total_vector_text.become(
                MathTex(rf"\vec{{F}}_{{\text{{tot}}}} = {fmt(f_tot)}", font_size=20, color=BLACK).move_to(self.total_vector_text)
            )

            if np.linalg.norm(f_tot) < 0.1:
                print(f"{f_drv = }")
                print(f"{f_rep = }")
                print(f"{f_tot = }")
                self.play(self.zero_force_notice.animate.set_opacity(1), run_time=0.3)
                self.wait(3)
                self.play(self.zero_force_notice.animate.set_opacity(0), run_time=0.3)
                count +=1
                if count ==2:
                    break
        
            if np.linalg.norm(self.agent_pos - self.goal_pos) < self.RADIUS + 0.2:
                self.play(Write(Text("Goal Reached!", font_size=40, color=GREEN)))
                self.wait(1)
                break
            self.wait(self.DT)

            # if step > steps/2:
            #     obstacle_velocity = np.array([0.0, -0.01, 0.0])
            #     self.obstacle.shift(obstacle_velocity)
            #     self.obstacle_label.shift(obstacle_velocity)
                
                
